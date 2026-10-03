#!/usr/bin/env python3
"""
Benchmark FoodDetector_inverted.mlpackage on all samples in data_balanced/test/.
Outputs results to outputs/benchmarks/benchmark_inverted_full.20260107.txt
Format matches outputs/benchmarks/benchmark_latest_full.txt
"""
import coremltools as ct
from pathlib import Path
import numpy as np
from PIL import Image
import time

model_path = Path("models/coreml/FoodDetector_inverted.mlpackage")
test_root = Path("data_balanced/test")
out_path = Path("outputs/benchmarks/benchmark_inverted_full.20260107.txt")

print(f"Loading model: {model_path}")
model = ct.models.MLModel(str(model_path))

# Find all test images
def find_test_images(root):
    return list(root.rglob('*.jpg')) + list(root.rglob('*.jpeg')) + list(root.rglob('*.png'))

imgs = find_test_images(test_root)
print(f"Found {len(imgs)} test images")

# Preprocess for FoodDetector: resize to 224x224, CHW, raw 0-255
def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    arr = arr[np.newaxis, ...]
    return arr

results = []
start_time = time.time()
for idx, img_path in enumerate(imgs, 1):
    arr = preprocess(img_path)
    input_dict = {'input': arr}
    t0 = time.time()
    out = model.predict(input_dict)
    t1 = time.time()
    food_prob = float(np.array(out['food_prob']).ravel()[0])
    not_food_prob = 1.0 - food_prob
    latency_ms = (t1 - t0) * 1000
    results.append((img_path, food_prob, not_food_prob, latency_ms))
    if idx % 100 == 0:
        print(f"Processed {idx}/{len(imgs)} images...")

elapsed = time.time() - start_time
print(f"Processed {len(imgs)} images in {elapsed:.1f}s")

# Write results
with open(out_path, 'w') as f:
    f.write("Runs: {}\n".format(len(results)))
    mean_latency = np.mean([r[3] for r in results])
    median_latency = np.median([r[3] for r in results])
    p95_latency = np.percentile([r[3] for r in results], 95)
    f.write(f"mean={mean_latency:.2f}ms  median={median_latency:.2f}ms  p95={p95_latency:.2f}ms\n\n")
    f.write("Sample classifications:\n")
    for idx, (img_path, food_prob, not_food_prob, latency_ms) in enumerate(results, 1):
        f.write(f"run#{idx:04d} {img_path}  time={latency_ms:.2f}ms  food_prob={food_prob:.4f}  not_food_prob={not_food_prob:.4f}\n")
print(f"Results written to {out_path}")
