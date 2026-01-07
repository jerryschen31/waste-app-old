#!/usr/bin/env python3
"""Benchmark Core ML models and run sample classifications.

Usage:
    ./coreml_env/bin/python3 scripts/benchmark_coreml.py

This script:
 - Loads one or more .mlpackage Core ML models
 - Runs warmup inferences
 - Runs timed inferences and reports median/mean/p95
 - Classifies a small random sample of images from `data_balanced/test` and prints results
"""

import argparse
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image
import coremltools as ct


def find_test_images(root: Path, max_images=2000):
    imgs = list(root.rglob('*.jpg')) + list(root.rglob('*.jpeg')) + list(root.rglob('*.png'))
    return imgs


def prepare_input(img_path: Path, spec, input_name: str, model_type: str = 'generic'):
    # Determine required shape
    desc = spec.description
    inp = desc.input[0]
    # Try to get multiArray shape
    shape = None
    try:
        shape = list(inp.type.multiArrayType.shape)
    except Exception:
        shape = None

    # Default target size
    if shape and len(shape) >= 3:
        # assume (3,H,W) or (1,3,H,W)
        if len(shape) == 4 and shape[0] == 1:
            _, c, h, w = shape
        elif len(shape) == 3:
            c, h, w = shape
        else:
            c, h, w = 3, 224, 224
    else:
        c, h, w = 3, 224, 224

    img = Image.open(img_path).convert('RGB')
    img = img.resize((w, h), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)

    # Heuristic: check input description for scale hints
    in_desc = ''
    try:
        in_desc = spec.description.input[0].shortDescription or ''
    except Exception:
        in_desc = ''

    if model_type == 'food_detector':
        # Food detector Core ML model expects 0..255 input (we traced wrapper dividing by 255)
        norm = arr
    else:
        if '255' in in_desc or '0,255' in in_desc:
            # model expects 0..255
            norm = arr
        else:
            # assume 0..1
            norm = arr / 255.0

    # Convert to CHW
    tensor = np.transpose(norm, (2, 0, 1)).astype(np.float32)
    # If spec indicates a batch dim (shape [1,3,H,W]), add batch dimension
    try:
        shape = list(inp.type.multiArrayType.shape)
        if len(shape) == 4 and shape[0] == 1:
            tensor = tensor[np.newaxis, ...]
    except Exception:
        pass
    return {input_name: tensor}


def detect_food_prob_fn(mlmodel, spec, imgs, input_name, samples_per_class=8):
    # Build a callable that maps raw model output to a food probability in [0,1].
    # Strategy: run a small calibration on labeled test images (food / not_food)
    # and decide whether the model returns a single sigmoid or a 2-class array
    # and which index corresponds to "food".
    food_imgs = [p for p in imgs if '/food/' in str(p) or '\\food\\' in str(p)]
    not_imgs = [p for p in imgs if '/not_food/' in str(p) or '\\not_food\\' in str(p)]
    if not food_imgs or not not_imgs:
        # fallback: assume single-value sigmoid where higher means more food
        return lambda outraw: float(np.array(outraw).ravel()[0])

    k = min(samples_per_class, len(food_imgs), len(not_imgs))
    rng = random.Random(0)
    sample_food = rng.sample(food_imgs, k)
    sample_not = rng.sample(not_imgs, k)

    def _read_out(p):
        inp = prepare_input(p, spec, input_name, model_type='food_detector')
        out = mlmodel.predict(inp)
        raw = out.get(spec.description.output[0].name)
        arr = np.array(raw).ravel()
        return arr

    vals_f = np.stack([_read_out(p) for p in sample_food])
    vals_n = np.stack([_read_out(p) for p in sample_not])

    # If outputs are scalar (shape (N,)) treat as sigmoid; else 2+ dims
    if vals_f.ndim == 1 or vals_f.shape[1] == 1:
        # scalar/sigmoid case: check polarity
        mean_f = float(vals_f.mean())
        mean_n = float(vals_n.mean())
        if mean_f >= mean_n:
            return lambda outraw: float(np.array(outraw).ravel()[0])
        else:
            return lambda outraw: 1.0 - float(np.array(outraw).ravel()[0])
    else:
        # multi-output case: find which index correlates with food samples
        mean_f = vals_f.mean(axis=0)
        mean_n = vals_n.mean(axis=0)
        # food index is the dimension where mean_f - mean_n is largest
        deltas = mean_f - mean_n
        food_idx = int(np.argmax(deltas))
        return lambda outraw: float(np.array(outraw).ravel()[food_idx])


def run_model(mlmodel, spec, imgs, model_type: str = 'generic', warmup=5, runs=50):
    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name

    # Warmup
    for i in range(min(warmup, len(imgs))):
        _inp = prepare_input(imgs[i], spec, input_name, model_type)
        mlmodel.predict(_inp)

    # Timed runs (reuse same image randomly)
    times = []
    rng = random.Random(0)
    for i in range(runs):
        img = imgs[rng.randrange(len(imgs))]
        inp = prepare_input(img, spec, input_name, model_type)
        t0 = time.perf_counter()
        out = mlmodel.predict(inp)
        t1 = time.perf_counter()
        times.append((t1 - t0, img, out.get(output_name)))

    ts = np.array([t for t, _, _ in times])
    return times, ts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['models/coreml/FoodDetector.mlpackage', 'models/coreml/RecyclingNet11.mlpackage'])
    p.add_argument('--num-samples', type=int, default=10)
    p.add_argument('--per-run', action='store_true', help='Print each individual run in sample classifications')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--runs', type=int, default=50)
    args = p.parse_args()

    test_root = Path('data_balanced/test')
    imgs = find_test_images(test_root)
    if not imgs:
        print('No test images found in data_balanced/test; aborting sample classification')
        imgs = []

    for mpath in args.models:
        mp = Path(mpath)
        if not mp.exists():
            print(f'Model not found: {mp} — skipping')
            continue
        print('\n' + '='*60)
        print(f'Loading Core ML model: {mp}')
        mlmodel = ct.models.MLModel(str(mp))
        spec = mlmodel.get_spec()

        # Heuristic model type detection by filename
        model_type = 'generic'
        if 'food' in mp.name.lower() or 'fooddector' in mp.name.lower() or 'fooddetector' in mp.name.lower():
            model_type = 'food_detector'
        if 'recycling' in mp.name.lower() or 'recyclingnet' in mp.name.lower():
            model_type = 'recycling_net'

        # Prepare model-specific helpers
        input_name = spec.description.input[0].name
        food_prob_fn = None
        if model_type == 'food_detector':
            # calibrate mapping from raw model output -> food probability
            food_prob_fn = detect_food_prob_fn(mlmodel, spec, imgs, input_name)

        # Sample inference + timings
        print('Running benchmark...')
        times, ts = run_model(mlmodel, spec, imgs, model_type=model_type, warmup=args.warmup, runs=args.runs)
        print(f'Runs: {len(ts)}  mean={ts.mean()*1000:.2f}ms  median={np.median(ts)*1000:.2f}ms  p95={np.percentile(ts,95)*1000:.2f}ms')

        # Show sample classifications for a few random images
        print('\nSample classifications:')
        if args.per_run:
            # Print each run separately (one line per timed run)
            for idx, (t, imgpath, outraw) in enumerate(times, start=1):
                if model_type == 'food_detector':
                    if food_prob_fn is not None:
                        prob = float(food_prob_fn(outraw))
                    else:
                        prob = float(np.array(outraw).ravel()[0]) if outraw is not None else float(outraw)
                    print(f'run#{idx:02d} {imgpath}  time={(t*1000):.2f}ms  food_prob={prob:.4f}  not_food_prob={(1-prob):.4f}')
                elif model_type == 'recycling_net':
                    o = np.array(outraw).ravel()
                    exps = np.exp(o - np.max(o))
                    probs = exps / exps.sum()
                    labels = [
                        'aluminium','batteries','cardboard','disposable plates','glass',
                        'hard plastic','paper','paper towel','polystyrene','soft plastics','takeaway cups'
                    ]
                    topk = np.argsort(-probs)[:3]
                    top = ', '.join([f'{labels[i]}:{probs[i]:.3f}' for i in topk])
                    agg_map = {
                        'aluminium':'Recycle','cardboard':'Recycle','glass':'Recycle','hard plastic':'Recycle','paper':'Recycle','soft plastics':'Recycle',
                        'batteries':'E-waste',
                        'paper towel':'Compost','disposable plates':'Compost','takeaway cups':'Compost',
                        'polystyrene':'Trash'
                    }
                    agg = {'Trash':0.0,'Compost':0.0,'Recycle':0.0,'E-waste':0.0,'Biological Waste':0.0}
                    for i, p in enumerate(probs):
                        lbl = labels[i]
                        cat = agg_map.get(lbl, 'Trash')
                        agg[cat] = agg.get(cat,0.0) + float(p)
                    agg_str = ', '.join([f'{k}:{v:.3f}' for k,v in agg.items()])
                    print(f'run#{idx:02d} {imgpath}  time={(t*1000):.2f}ms  top3=[{top}]  aggregated=[{agg_str}]')
                else:
                    print(f'run#{idx:02d} {imgpath}  time={(t*1000):.2f}ms  out={outraw}')
        else:
            rng = random.Random(1)
            samples = rng.sample(times, min(args.num_samples, len(times)))
            for t, imgpath, outraw in samples:
                # For FoodDetector: single sigmoid probability
                if model_type == 'food_detector':
                    if food_prob_fn is not None:
                        prob = float(food_prob_fn(outraw))
                    else:
                        prob = float(np.array(outraw).ravel()[0]) if outraw is not None else float(outraw)
                    print(f"{imgpath}  time={(t*1000):.2f}ms  food_prob={prob:.4f}  not_food_prob={(1-prob):.4f}")
                elif model_type == 'recycling_net':
                    o = np.array(outraw).ravel()
                    # compute softmax probabilities
                    exps = np.exp(o - np.max(o))
                    probs = exps / exps.sum()
                    # label mapping consistent with Sources/RecycleMVPKit/ModelLoader.swift
                    labels = [
                        'aluminium','batteries','cardboard','disposable plates','glass',
                        'hard plastic','paper','paper towel','polystyrene','soft plastics','takeaway cups'
                    ]
                    topk = np.argsort(-probs)[:3]
                    top = ', '.join([f'{labels[i]}:{probs[i]:.3f}' for i in topk])

                    # aggregate into 5 disposal categories
                    agg_map = {
                        'aluminium':'Recycle','cardboard':'Recycle','glass':'Recycle','hard plastic':'Recycle','paper':'Recycle','soft plastics':'Recycle',
                        'batteries':'E-waste',
                        'paper towel':'Compost','disposable plates':'Compost','takeaway cups':'Compost',
                        'polystyrene':'Trash'
                    }
                    agg = {'Trash':0.0,'Compost':0.0,'Recycle':0.0,'E-waste':0.0,'Biological Waste':0.0}
                    for i, p in enumerate(probs):
                        lbl = labels[i]
                        cat = agg_map.get(lbl, 'Trash')
                        agg[cat] = agg.get(cat,0.0) + float(p)

                    agg_str = ', '.join([f'{k}:{v:.3f}' for k,v in agg.items()])
                    print(f"{imgpath}  time={(t*1000):.2f}ms  top3=[{top}]  aggregated=[{agg_str}]")
                else:
                    # generic fallback: print raw out
                    if isinstance(outraw, (list, tuple, np.ndarray)):
                        prob_str = np.array(outraw).ravel()[:3]
                        print(f'{imgpath}  time={(t*1000):.2f}ms  out_top3={prob_str}')
                    else:
                        print(f'{imgpath}  time={(t*1000):.2f}ms  out={outraw}')


if __name__ == '__main__':
    main()
