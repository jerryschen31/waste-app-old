#!/usr/bin/env python3
"""
Benchmark FoodDetector model on data_balanced/test/ dataset (all 2000 images).
Outputs results to outputs/benchmarks/benchmark_latest_full.20260107.txt
Format matches outputs/benchmarks/benchmark_latest_full.txt
"""

import os
import json
import logging
import random
from pathlib import Path
from collections import defaultdict
import time
import numpy as np
from PIL import Image
import coremltools as ct

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_image_fd(image_path, target_size=224):
    """Preprocess image for FoodDetector (224x224, no normalization - expects raw 0-255)."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy array in CHW format (raw 0-255, no normalization)
    img_array = np.array(img, dtype=np.float32)  # (224, 224, 3), 0-255
    img_array = np.transpose(img_array, (2, 0, 1))  # CHW format (3, 224, 224)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # (1, 3, 224, 224)
    return img_array.astype(np.float32)

def detect_food_prob_fn(model, imgs, samples_per_class=8):
    """Calibrate output interpretation by testing on labeled food/not_food samples.
    
    Returns a callable that maps raw model output to a food probability in [0,1].
    Handles inverted polarity automatically.
    """
    food_imgs = [p for p in imgs if '/food/' in str(p) or '\\food\\' in str(p)]
    not_imgs = [p for p in imgs if '/not_food/' in str(p) or '\\not_food\\' in str(p)]
    
    if not food_imgs or not not_imgs:
        # Fallback: assume raw output is food probability
        return lambda outraw: float(np.array(outraw).ravel()[0])
    
    k = min(samples_per_class, len(food_imgs), len(not_imgs))
    rng = np.random.RandomState(0)
    sample_food = rng.choice(food_imgs, k, replace=False).tolist()
    sample_not = rng.choice(not_imgs, k, replace=False).tolist()
    
    def _read_out(p):
        """Read model output for an image."""
        img_array = preprocess_image_fd(p)
        inp = {'input': img_array}
        out = model.predict(inp)
        raw = out.get('var_1626')
        return float(np.array(raw).ravel()[0])
    
    vals_f = np.array([_read_out(p) for p in sample_food])
    vals_n = np.array([_read_out(p) for p in sample_not])
    
    mean_f = float(vals_f.mean())
    mean_n = float(vals_n.mean())
    
    # If food mean >= not_food mean, use raw value; else invert
    if mean_f >= mean_n:
        return lambda outraw: float(np.array(outraw).ravel()[0])
    else:
        return lambda outraw: 1.0 - float(np.array(outraw).ravel()[0])

def run_benchmark():
    """Run FoodDetector benchmark on all balanced test images."""
    
    # Paths
    workspace_root = Path(__file__).parent.parent
    model_path = workspace_root / "models" / "coreml" / "FoodDetector.mlpackage"
    test_root = workspace_root / "data_balanced" / "test"
    out_path = workspace_root / "outputs" / "benchmarks" / "benchmark_latest_full.20260107.txt"
    
    logger.info("=" * 60)
    logger.info("FOOD DETECTOR BENCHMARK")
    logger.info("=" * 60)
    
    # Check paths
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    if not test_root.exists():
        logger.error(f"Test directory not found: {test_root}")
        return
    
    # Load model
    logger.info(f"Loading FoodDetector from {model_path}...")
    model = ct.models.MLModel(str(model_path))
    logger.info("Model loaded successfully")
    
    # Find all test images
    test_root_resolved = test_root.resolve()
    all_imgs = sorted(list(test_root_resolved.rglob('*.jpg')) + list(test_root_resolved.rglob('*.png')))
    
    food_imgs = [p for p in all_imgs if '/food/' in str(p) or '\\food\\' in str(p)]
    not_imgs = [p for p in all_imgs if '/not_food/' in str(p) or '\\not_food\\' in str(p)]
    
    logger.info(f"Found {len(food_imgs)} food images")
    logger.info(f"Found {len(not_imgs)} not_food images")
    logger.info(f"Total test images: {len(all_imgs)}")
    
    # Calibrate output interpretation
    logger.info("\nCalibrating output interpretation...")
    food_prob_fn = detect_food_prob_fn(model, all_imgs)
    logger.info("Calibration complete")
    
    logger.info("\nRunning inference on all {} images...".format(len(all_imgs)))
    
    times = []
    latencies = []
    run_num = 0
    
    for img_path in all_imgs:
        try:
            # Preprocess
            image_array = preprocess_image_fd(str(img_path))
            
            # Infer with timing
            start = time.time()
            input_dict = {'input': image_array}
            output = model.predict(input_dict)
            t_end = time.time()
            latency = (t_end - start) * 1000  # ms
            latencies.append(latency)
            
            # Extract and calibrate confidence
            raw_confidence = output.get('var_1626')
            confidence = food_prob_fn(raw_confidence)
            not_food_prob = 1.0 - confidence
            
            times.append((latency, img_path, confidence, not_food_prob))
            run_num += 1
            
            if run_num % 200 == 0:
                logger.info(f"  Processed {run_num}/{len(all_imgs)} images...")
        
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue
    
    logger.info(f"  Processed {run_num}/{len(all_imgs)} images")
    
    # Calculate statistics
    ts = np.array(latencies)
    mean_latency = ts.mean()
    median_latency = np.median(ts)
    p95_latency = np.percentile(ts, 95)
    
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"\nRuns: {len(times)}  mean={mean_latency:.2f}ms  median={median_latency:.2f}ms  p95={p95_latency:.2f}ms")
    logger.info("\nSample classifications:")
    
    # Write results to file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("Loading Core ML model: models/coreml/FoodDetector.mlpackage\n")
        f.write("Running benchmark...\n")
        f.write(f"Runs: {len(times)}  mean={mean_latency:.2f}ms  median={median_latency:.2f}ms  p95={p95_latency:.2f}ms\n")
        f.write("\n")
        f.write("Sample classifications:\n")
        
        for idx, (latency, img_path, food_prob, not_food_prob) in enumerate(times, 1):
            rel_path = img_path.relative_to(workspace_root)
            line = f"run#{idx:04d} {rel_path}  time={latency:.2f}ms  food_prob={food_prob:.4f}  not_food_prob={not_food_prob:.4f}\n"
            f.write(line)
    
    logger.info(f"\nResults written to: {out_path}")
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 60)

if __name__ == '__main__':
    run_benchmark()
