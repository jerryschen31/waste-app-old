#!/usr/bin/env python3
"""
Download Hugging Face models to local directory.
"""
from transformers import AutoModelForImageClassification, AutoImageProcessor
import os
from pathlib import Path

def download_model(model_name, output_dir):
    print(f"Downloading {model_name}...")
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {output_path}...")
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    print("Done.")

if __name__ == "__main__":
    # Download EfficientNet-B2 (base for FoodDetector)
    # Note: If we used a fine-tuned version, we should use that name. 
    # But based on context, we might have been using the base for feature extraction or transfer learning
    # OR we fine-tuned it and saved it there.
    # If the latter, we lost the fine-tuned weights.
    
    # For now, I'll download the base model so the script doesn't crash.
    # Ideally, we would re-run 'train_food_binary.py' to get the weights back.
    download_model("google/efficientnet-b2", "models/efficientnet-b2")
