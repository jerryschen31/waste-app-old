#!/usr/bin/env python3
"""
Benchmark the Fine-Tuned Cascade Model Pipeline.

This script runs the full cascade pipeline:
1. FoodDetector (Core ML) -> Checks for Compost (biological)
2. RecyclingNet11ex (Core ML) -> Fine-tuned model for other categories

Optimized for the new 11-class structure where 'electronics' encompasses e-waste.
"""

import coremltools as ct
from PIL import Image
import os
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import json

# Configuration
TEST_DATA_DIR = Path("data/waste_test_filtered")
FOOD_MODEL_PATH = "models/coreml/FoodDetector.mlpackage"
# Use the fine-tuned model
RECYCLING_MODEL_PATH = "models/coreml/RecyclingNet11ex.mlpackage" 
OUTPUT_DIR = Path("outputs/benchmarks/run-20260121-6")

# Mapping: Fine-tuned Model Classes -> Final User Categories
# Fine-tuned classes: ['biological', 'cardboard', 'clothes', 'electronics', 'glass', 'metal', 'other', 'paper', 'plastic', 'shoes', 'trash']
RN_TO_FINAL_CATEGORY = {
    'biological': 'Compost',   # Should be caught by FoodDetector, but fallback
    'cardboard': 'Recycle',
    'clothes': 'Trash',        # Changed from Recycle
    'electronics': 'e-Waste',  # NEW category
    'glass': 'Recycle',
    'metal': 'Recycle',
    'other': 'Trash',          # Changed from Recycle/Other
    'paper': 'Recycle',
    'plastic': 'Recycle',
    'shoes': 'Trash',          # Changed from Recycle
    'trash': 'Trash'
}

def load_models():
    """Load both Core ML models."""
    print(f"Loading FoodDetector from {FOOD_MODEL_PATH}...")
    food_model = ct.models.MLModel(str(FOOD_MODEL_PATH))
    
    print(f"Loading RecyclingNet11ex from {RECYCLING_MODEL_PATH}...")
    recycling_model = ct.models.MLModel(str(RECYCLING_MODEL_PATH))
    
    return food_model, recycling_model

def get_recycling_prediction(model, image):
    """Get prediction from RecyclingNet11ex."""
    # Core ML expects PIL Image for ImageType inputs
    try:
        prediction = model.predict({'input': image})
        
        # The export script used 'classLabel' or just probabilities
        # We need to handle the output format of our exported model
        # The export script output: 10-class probabilities (Softmax) dictionary or array
        
        # Inspect prediction keys to adapt
        if 'classLabel' in prediction:
            return prediction['classLabel'], prediction.get('classLabel_probs', {})
            
        # If output is just probabilities (common in custom exports)
        # Find the max probability
        for key, value in prediction.items():
            # Assume the dictionary output is the probabilities
            if isinstance(value, dict):
                probs = value
                best_class = max(probs, key=probs.get)
                return best_class, probs
            
        # Fallback if raw array (requires class names knowledge)
        # This script assumes dictionary output with class names
        print(f"Warning: Unexpected output format: {prediction.keys()}")
        return "unknown", {}
        
    except Exception as e:
        print(f"Error predicting: {e}")
        return "error", {}

def run_benchmark():
    """Run the benchmark on test data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    food_model, recycling_model = load_models()
    
    results = []
    latencies = []
    
    # Files for detailed reporting
    f_summary = open(OUTPUT_DIR / "SUMMARY.txt", "w")
    f_food = open(OUTPUT_DIR / "food_model_benchmark.txt", "w")
    f_full = open(OUTPUT_DIR / "full_model_benchmark.txt", "w")
    f_misc = open(OUTPUT_DIR / "SUMMARY_MISCLASSIFIED.txt", "w")
    
    misclassified_md = ["# Misclassification Analysis\n\n| Image | True Label | Predicted | Food Conf | RN Class | RN Conf |\n|---|---|---|---|---|---|\n"]
    
    # Walk through test directory
    files = []
    for category in TEST_DATA_DIR.iterdir():
        if category.is_dir():
            for img_path in category.glob("*.jpg"):
                files.append((category.name, img_path))
            for img_path in category.glob("*.jpeg"):
                files.append((category.name, img_path))
            for img_path in category.glob("*.png"):
                files.append((category.name, img_path))
    
    print(f"Benchmarking {len(files)} images...")
    
    correct_count = 0
    category_stats = {
        'Compost': {'total': 0, 'correct': 0},
        'Recycle': {'total': 0, 'correct': 0},
        'Trash': {'total': 0, 'correct': 0},
        'e-Waste': {'total': 0, 'correct': 0},
        'Hazardous': {'total': 0, 'correct': 0}, # If any
        'Other': {'total': 0, 'correct': 0}      # If any
    }
    
    for true_category, img_path in tqdm(files):
        # Normalize true category names if needed
        if true_category == "e_waste": true_category = "e-Waste"
        if true_category == "compost": true_category = "Compost"
        if true_category == "recycle": true_category = "Recycle"
        if true_category == "trash": true_category = "Trash"
        
        # Track total
        if true_category not in category_stats:
            category_stats[true_category] = {'total': 0, 'correct': 0}
        category_stats[true_category]['total'] += 1
        
        try:
            image = Image.open(img_path).convert('RGB')
            # Resize is handled by Core ML, but ensuring input is standard helps
            image = image.resize((224, 224))
            
            start_time = time.time()
            
            # --- STAGE 1: Food Detector ---
            food_pred = food_model.predict({'input': image})
            
            # Access probabilities (key might vary, usually 'classLabel_probs' or similar)
            # Assuming standard Core ML classifier export
            probs = list(food_pred.values())[0] # simplistic access
            if isinstance(probs, dict):
                food_prob = probs.get('food', 0.0)
            else:
                # If the user exported custom names or inverted
                # You might need to adjust this based on FoodDetector export keys
                # For now assuming 'food' is the key
                 food_prob = 0.0 # Placeholder if key not found
                 for k, v in food_pred.items():
                     if isinstance(v, dict) and 'food' in v:
                         food_prob = v['food']

            
            is_food = food_prob > 0.5  # Threshold
            
            f_food.write(f"{img_path.name}: Food Prob {food_prob:.4f} -> {'Food' if is_food else 'Non-Food'}\n")
            
            final_pred = "Unknown"
            rn_class = "N/A"
            rn_conf = 0.0
            
            if is_food:
                final_pred = "Compost"
            else:
                # --- STAGE 2: RecyclingNet ---
                rn_class, rn_probs = get_recycling_prediction(recycling_model, image)
                rn_conf = rn_probs.get(rn_class, 0.0) if rn_probs else 0.0
                
                # Map to final category
                final_pred = RN_TO_FINAL_CATEGORY.get(rn_class, "Trash")
            
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)
            
            # Check accuracy
            is_correct = (final_pred.lower() == true_category.lower())
            
            if is_correct:
                correct_count += 1
                category_stats[true_category]['correct'] += 1
            else:
                # Log misclassification
                error_msg = f"{img_path.name}: True={true_category}, Pred={final_pred} (Food={food_prob:.2f}, RN={rn_class} {rn_conf:.2f})"
                f_misc.write(error_msg + "\n")
                misclassified_md.append(f"| {img_path.name} | {true_category} | **{final_pred}** | {food_prob:.2f} | {rn_class} | {rn_conf:.2f} |")
            
            f_full.write(f"{img_path.name},{true_category},{final_pred},{is_correct},{food_prob:.4f},{rn_class},{rn_conf:.4f}\n")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    # Calculate stats
    total_latencies = np.array(latencies)
    latency_stats = {
        'mean': np.mean(total_latencies),
        'median': np.median(total_latencies),
        'p95': np.percentile(total_latencies, 95)
    }
    
    accuracy = (correct_count / len(files)) * 100
    
    # Write Summary
    f_summary.write("==================================================\n")
    f_summary.write("BENCHMARK SUMMARY (Fine-Tuned Cascade)\n")
    f_summary.write("==================================================\n")
    f_summary.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f_summary.write(f"Model: {RECYCLING_MODEL_PATH}\n")
    f_summary.write(f"Total Images: {len(files)}\n")
    f_summary.write(f"Overall Accuracy: {accuracy:.2f}% ({correct_count}/{len(files)})\n\n")
    
    f_summary.write("Category Performance:\n")
    f_summary.write("---------------------\n")
    for cat, stats in category_stats.items():
        if stats['total'] > 0:
            cat_acc = (stats['correct'] / stats['total']) * 100
            f_summary.write(f"{cat:10s}: {cat_acc:6.2f}% ({stats['correct']}/{stats['total']})\n")
    
    f_summary.write("\nLatency (ms):\n")
    f_summary.write(f"Mean:   {latency_stats['mean']:.2f}\n")
    f_summary.write(f"Median: {latency_stats['median']:.2f}\n")
    f_summary.write(f"P95:    {latency_stats['p95']:.2f}\n")
    
    # Save markdown report
    with open(OUTPUT_DIR / "MISCLASSIFIED_IMAGES.md", "w") as f_md:
        f_md.write("\n".join(misclassified_md))
        
    f_summary.close()
    f_food.close()
    f_full.close()
    f_misc.close()
    
    print("\nBenchmark Complete!")
    print(f"Summary saved to {OUTPUT_DIR}/SUMMARY.txt")
    print(f"Overall Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    run_benchmark()
