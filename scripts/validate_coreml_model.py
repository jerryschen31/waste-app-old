#!/usr/bin/env python3
"""
Validate the exported Core ML model.

This script loads the .mlpackage and tests it on a few sample images
to ensure the export process was successful and outputs are correct.

Usage:
    python3 scripts/validate_coreml_model.py
"""

import coremltools as ct
from PIL import Image
from pathlib import Path
import sys
import numpy as np

# Configuration
MODEL_PATH = "models/coreml/RecyclingNet11ex.mlpackage"
TEST_IMAGE_DIR = Path("data_finetune/val/electronics") # Use validation images

def validate_model():
    print(f"Loading model from {MODEL_PATH}...")
    
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        sys.exit(1)
        
    try:
        model = ct.models.MLModel(MODEL_PATH)
        print("✅ Model loaded successfully")
        
        # Print model metadata
        print(f"Description: {model.short_description}")
        print(f"Input description: {model.input_description}")
        print(f"Output description: {model.output_description}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Test on images
    test_images = list(TEST_IMAGE_DIR.glob("*.jpg"))[:3]
    if not test_images:
        print(f"⚠️ No test images found in {TEST_IMAGE_DIR}")
        return

    print("\nRunning inference tests...")
    print("-" * 50)
    
    for img_path in test_images:
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize((224, 224))
            
            prediction = model.predict({'input': image})
            
            print(f"Image: {img_path.name}")
            
            # Parse prediction output
            # Assuming standard classifier output or dictionary of probs
            class_label = prediction.get('classLabel')
            probs = prediction.get('classLabel_probs') # Standard key for classifier
            
            # If not standard classifier, try to interpret
            if not probs:
                 # Check if the output is just the prob dictionary directly
                 for k, v in prediction.items():
                     if isinstance(v, dict):
                         probs = v
                         # Find max
                         if not class_label:
                             class_label = max(probs, key=probs.get)
            
            if class_label and probs:
                confidence = probs.get(class_label, 0.0)
                print(f"  Prediction: {class_label}")
                print(f"  Confidence: {confidence:.2%}")
                
                 # Check if e-waste
                if class_label == "electronics":
                    print("  ✅ Correctly identified as electronics")
                else:
                    print(f"  ⚠️  classified as {class_label}")

            elif probs: # Just probs
                 best_class = max(probs, key=probs.get)
                 print(f"  Prediction: {best_class} (from probs)")
            else:
                print(f"  Raw Output: {prediction.keys()}")

            print("-" * 50)

        except Exception as e:
            print(f"❌ Error predicting {img_path.name}: {e}")

if __name__ == "__main__":
    validate_model()
