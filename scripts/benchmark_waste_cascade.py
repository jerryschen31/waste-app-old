#!/usr/bin/env python3
"""
Benchmark the waste classification cascade on test dataset.

Runs FoodDetector → RecyclingNet11 cascade on all images in data/waste_test/
and generates per-category accuracy and latency reports.
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
import sys

import coremltools as ct
from PIL import Image
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model paths
FOOD_DETECTOR_PATH = Path('/Users/jerry/gh/waste-app/models/coreml/FoodDetector.mlpackage')
RECYCLING_NET_PATH = Path('/Users/jerry/gh/waste-app/models/coreml/RecyclingNet11.mlpackage')
TEST_DATA_ROOT = Path('/Users/jerry/gh/waste-app/data/waste_test')

# Category mappings
WASTE_CATEGORIES = ['trash', 'recycle', 'compost', 'e_waste', 'other']

# RecyclingNet11 class labels (11 classes → 5 disposal categories)
RECYCLING_NET_LABELS = [
    'aluminium', 'batteries', 'cardboard', 'disposable plates',
    'glass', 'hard plastic', 'paper', 'paper towel',
    'polystyrene', 'soft plastics', 'takeaway cups'
]

# Mapping from 11-class to 5-category disposal
CLASS_TO_DISPOSAL = {
    'aluminium': 'recycle',
    'batteries': 'e_waste',
    'cardboard': 'recycle',
    'disposable plates': 'compost',
    'glass': 'recycle',
    'hard plastic': 'recycle',
    'paper': 'recycle',
    'paper towel': 'compost',
    'polystyrene': 'trash',
    'soft plastics': 'recycle',
    'takeaway cups': 'compost'
}


class CascadeClassifier:
    """Waste classification cascade: FoodDetector → RecyclingNet11."""
    
    def __init__(self):
        self.food_detector = None
        self.recycling_net = None
        self.load_models()
    
    def load_models(self):
        """Load both Core ML models."""
        logger.info("Loading FoodDetector...")
        try:
            self.food_detector = ct.models.MLModel(str(FOOD_DETECTOR_PATH))
        except Exception as e:
            logger.error(f"Failed to load FoodDetector: {e}")
            raise
        
        logger.info("Loading RecyclingNet11...")
        try:
            self.recycling_net = ct.models.MLModel(str(RECYCLING_NET_PATH))
        except Exception as e:
            logger.error(f"Failed to load RecyclingNet11: {e}")
            raise
    
    def preprocess_image(self, image_path: Path, size: int = 384) -> np.ndarray:
        """Load and preprocess image to tensor."""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            
            # Convert to numpy array (values in 0-255)
            arr = np.array(img, dtype=np.float32)
            
            # Add batch and channel dimensions
            arr = np.expand_dims(arr, axis=0)  # (H, W, 3) → (1, H, W, 3)
            arr = np.transpose(arr, (0, 3, 1, 2))  # (1, H, W, 3) → (1, 3, H, W)
            
            return arr
        except Exception as e:
            logger.error(f"Failed to preprocess {image_path}: {e}")
            return None
    
    def infer_food_detector(self, image_array: np.ndarray) -> tuple:
        """Run FoodDetector inference. Returns (is_food_prob, is_food)."""
        try:
            # FoodDetector expects input (1,3,224,224)
            # Resize if needed
            if image_array.shape[2:] != (224, 224):
                img_pil = Image.fromarray(np.transpose(image_array[0], (1, 2, 0)).astype(np.uint8))
                img_pil = img_pil.resize((224, 224), Image.Resampling.LANCZOS)
                arr = np.array(img_pil, dtype=np.float32)
                arr = np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)
                image_array = arr
            
            input_dict = {'input': image_array.astype(np.float16)}
            output = self.food_detector.predict(input_dict)
            
            # Extract food probability
            food_prob_array = output.get('var_1626', np.array([[0.5]]))
            if isinstance(food_prob_array, dict):
                food_prob_array = food_prob_array.get('data', np.array([[0.5]]))
            
            # Flatten to scalar
            food_prob = float(np.array(food_prob_array).flatten()[0])
            is_food = food_prob > 0.5
            
            return food_prob, is_food
        except Exception as e:
            logger.debug(f"FoodDetector inference error: {e}")
            return 0.5, False
    
    def infer_recycling_net(self, image_array: np.ndarray) -> tuple:
        """Run RecyclingNet11 inference. Returns (disposal_class, confidence)."""
        try:
            # RecyclingNet expects input (1,3,384,384)
            if image_array.shape[2:] != (384, 384):
                img_pil = Image.fromarray(np.transpose(image_array[0], (1, 2, 0)).astype(np.uint8))
                img_pil = img_pil.resize((384, 384), Image.Resampling.LANCZOS)
                arr = np.array(img_pil, dtype=np.float32)
                arr = np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)
                image_array = arr
            
            input_dict = {'input': image_array.astype(np.float16)}
            output = self.recycling_net.predict(input_dict)
            
            # Extract logits
            logits_array = output.get('var_760', np.zeros((1, 11)))
            if isinstance(logits_array, dict):
                logits_array = logits_array.get('data', np.zeros((1, 11)))
            
            logits = np.array(logits_array, dtype=np.float32).flatten()
            probs = self._softmax(logits)
            class_idx = np.argmax(probs)
            confidence = float(probs[class_idx])
            
            class_name = RECYCLING_NET_LABELS[class_idx]
            disposal = CLASS_TO_DISPOSAL.get(class_name, 'trash')
            
            return disposal, confidence, class_name
        except Exception as e:
            logger.debug(f"RecyclingNet11 inference error: {e}")
            return 'trash', 0.0, 'unknown'
    
    def classify(self, image_path: Path) -> dict:
        """Run cascade classification on image."""
        img_array = self.preprocess_image(image_path)
        if img_array is None:
            return None
        
        start_time = time.time()
        
        # Step 1: FoodDetector
        food_prob, is_food = self.infer_food_detector(img_array)
        
        if is_food:
            # Food detected → return Compost directly
            result = {
                'image': image_path.name,
                'stage': 'food_detector',
                'prediction': 'compost',
                'confidence': float(food_prob),
                'raw_class': 'food',
                'latency_ms': (time.time() - start_time) * 1000
            }
        else:
            # Not food → run RecyclingNet11
            disposal, conf, raw_class = self.infer_recycling_net(img_array)
            result = {
                'image': image_path.name,
                'stage': 'recycling_net',
                'prediction': disposal,
                'confidence': float(conf),
                'raw_class': raw_class,
                'latency_ms': (time.time() - start_time) * 1000
            }
        
        return result
    
    @staticmethod
    def _softmax(logits):
        """Compute softmax from logits."""
        if isinstance(logits, dict):
            logits = logits.get('data', np.zeros(2))
        
        logits = np.array(logits, dtype=np.float64)
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum(axis=0)


def main():
    logger.info("="*70)
    logger.info("WASTE CLASSIFICATION CASCADE BENCHMARK")
    logger.info("="*70)
    
    # Initialize cascade
    cascade = CascadeClassifier()
    
    # Collect all test images
    test_images = []
    ground_truth = {}
    
    for category in WASTE_CATEGORIES:
        cat_dir = TEST_DATA_ROOT / category
        if not cat_dir.exists():
            logger.warning(f"Category directory not found: {cat_dir}")
            continue
        
        images = list(cat_dir.glob('*.jpg'))
        test_images.extend(images)
        for img in images:
            ground_truth[img.name] = category
    
    logger.info(f"Found {len(test_images)} test images across {len(WASTE_CATEGORIES)} categories")
    
    # Run inference on all images
    results = []
    latencies = []
    predictions_by_category = defaultdict(lambda: defaultdict(int))
    
    logger.info("\nRunning inference...")
    for idx, image_path in enumerate(test_images):
        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{len(test_images)} images...")
        
        result = cascade.classify(image_path)
        if result is None:
            continue
        
        results.append(result)
        latencies.append(result['latency_ms'])
        
        true_category = ground_truth[image_path.name]
        predicted_category = result['prediction']
        predictions_by_category[true_category][predicted_category] += 1
    
    logger.info(f"  Processed {len(results)}/{len(test_images)} images")
    
    # Compute accuracy metrics
    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)
    
    correct = 0
    per_category_accuracy = {}
    
    for true_cat in WASTE_CATEGORIES:
        total = sum(predictions_by_category[true_cat].values())
        if total == 0:
            continue
        
        correct_in_cat = predictions_by_category[true_cat][true_cat]
        acc = correct_in_cat / total * 100
        per_category_accuracy[true_cat] = acc
        correct += correct_in_cat
        
        logger.info(f"\n{true_cat.upper()}: {acc:.1f}% ({correct_in_cat}/{total})")
        
        # Show confusion matrix row
        for pred_cat in WASTE_CATEGORIES:
            count = predictions_by_category[true_cat][pred_cat]
            if count > 0:
                pct = count / total * 100
                logger.info(f"  → {pred_cat}: {count:3d} ({pct:5.1f}%)")
    
    overall_accuracy = correct / len(results) * 100 if results else 0
    
    logger.info(f"\n{'='*70}")
    logger.info(f"OVERALL ACCURACY: {overall_accuracy:.1f}% ({correct}/{len(results)})")
    logger.info(f"{'='*70}")
    
    # Latency statistics
    latencies_ms = [r['latency_ms'] for r in results]
    logger.info(f"\nLATENCY STATISTICS:")
    logger.info(f"  Mean:   {np.mean(latencies_ms):.1f} ms")
    logger.info(f"  Median: {np.median(latencies_ms):.1f} ms")
    logger.info(f"  Stdev:  {np.std(latencies_ms):.1f} ms")
    logger.info(f"  Min:    {np.min(latencies_ms):.1f} ms")
    logger.info(f"  Max:    {np.max(latencies_ms):.1f} ms")
    
    # Stage breakdown
    food_detector_count = sum(1 for r in results if r['stage'] == 'food_detector')
    recycling_net_count = sum(1 for r in results if r['stage'] == 'recycling_net')
    logger.info(f"\nPIPELINE BREAKDOWN:")
    logger.info(f"  Food Detector (skipped RecyclingNet): {food_detector_count} ({food_detector_count/len(results)*100:.1f}%)")
    logger.info(f"  RecyclingNet (ran full inference):    {recycling_net_count} ({recycling_net_count/len(results)*100:.1f}%)")
    
    # Save detailed results
    output_file = Path('/Users/jerry/gh/waste-app/outputs/benchmark_waste_cascade.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    benchmark_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_images': len(results),
        'overall_accuracy': overall_accuracy,
        'per_category_accuracy': per_category_accuracy,
        'latency_stats': {
            'mean_ms': float(np.mean(latencies_ms)),
            'median_ms': float(np.median(latencies_ms)),
            'stdev_ms': float(np.std(latencies_ms)),
            'min_ms': float(np.min(latencies_ms)),
            'max_ms': float(np.max(latencies_ms))
        },
        'pipeline_breakdown': {
            'food_detector_count': food_detector_count,
            'recycling_net_count': recycling_net_count
        },
        'confusion_matrix': {
            true_cat: {pred_cat: predictions_by_category[true_cat][pred_cat] 
                      for pred_cat in WASTE_CATEGORIES}
            for true_cat in WASTE_CATEGORIES
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    logger.info(f"\nDetailed results saved to: {output_file}")
    
    # Save per-image results
    results_file = Path('/Users/jerry/gh/waste-app/outputs/benchmark_waste_cascade_detailed.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Per-image results saved to: {results_file}")
    
    logger.info("\n" + "="*70)
    logger.info("BENCHMARK COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    main()
