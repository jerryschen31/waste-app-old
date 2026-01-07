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


def prepare_input(img_path: Path, spec, input_name: str):
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


def run_model(mlmodel, spec, imgs, warmup=5, runs=50):
    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name

    # Warmup
    for i in range(min(warmup, len(imgs))):
        _inp = prepare_input(imgs[i], spec, input_name)
        mlmodel.predict(_inp)

    # Timed runs (reuse same image randomly)
    times = []
    rng = random.Random(0)
    for i in range(runs):
        img = imgs[rng.randrange(len(imgs))]
        inp = prepare_input(img, spec, input_name)
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

        # Sample inference + timings
        print('Running benchmark...')
        times, ts = run_model(mlmodel, spec, imgs, warmup=args.warmup, runs=args.runs)
        print(f'Runs: {len(ts)}  mean={ts.mean()*1000:.2f}ms  median={np.median(ts)*1000:.2f}ms  p95={np.percentile(ts,95)*1000:.2f}ms')

        # Show sample classifications for a few random images
        print('\nSample classifications:')
        rng = random.Random(1)
        samples = rng.sample(times, min(args.num_samples, len(times)))
        for t, imgpath, outprob in samples:
            # outprob may be array-like; print a compact summary
            if isinstance(outprob, (list, tuple, np.ndarray)):
                o = np.array(outprob).ravel()
                topk = np.argsort(-o)[:3]
                top = ', '.join([f'{i}:{o[i]:.3f}' for i in topk])
                prob_str = f'[{top}]'
            else:
                prob_str = f'{float(outprob):.4f}'
            print(f'{imgpath}  time={(t*1000):.2f}ms  out={prob_str}')


if __name__ == '__main__':
    main()
