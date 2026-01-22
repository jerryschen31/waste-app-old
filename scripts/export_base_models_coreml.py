#!/usr/bin/env python3
"""
Export RecyclingNet11 (original) and FoodDetector models to Core ML format.

This script converts the pre-trained models to Core ML (.mlpackage) format.

Usage:
    python3 scripts/export_base_models_coreml.py
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification
import coremltools as ct

# Model paths
RECYCLING_NET_DIR = Path("models/recycling-net-11")
EFFICIENTNET_DIR = Path("models/efficientnet-b2")
OUTPUT_DIR = Path("models/coreml")

def find_model_snapshot(base_dir):
    """Find the model snapshot directory."""
    snapshots = list(base_dir.glob("models--*"))
    if snapshots:
        return list(snapshots[0].glob("snapshots/*"))[0]
    return None

def export_recycling_net():
    """Export RecyclingNet11 (original) to Core ML."""
    print("\n" + "=" * 70)
    print("EXPORTING RECYCLINGNET11 (ORIGINAL)")
    print("=" * 70)
    
    model_path = find_model_snapshot(RECYCLING_NET_DIR)
    if not model_path:
        print("❌ RecyclingNet11 model not found")
        return
    
    print(f"Loading from: {model_path}")
    
    # Load model and processor
    model = AutoModelForImageClassification.from_pretrained(str(model_path))
    processor = AutoImageProcessor.from_pretrained(str(model_path))
    
    # Get class names from config
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    class_names = config.get("id2label", {})
    if isinstance(class_names, dict):
        class_names = [class_names.get(str(i), f"class_{i}") for i in range(len(class_names))]
    
    print(f"Classes: {len(class_names)}")
    
    # Get input size
    if hasattr(processor, 'size'):
        size_dict = processor.size
        if isinstance(size_dict, dict):
            height = size_dict.get('height', 224)
            width = size_dict.get('width', 224)
        else:
            height = width = size_dict
    else:
        height = width = 224
    
    print(f"Input size: {height}x{width}")
    
    # Create wrapper with Softmax
    class ModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.softmax = torch.nn.Softmax(dim=1)
        
        def forward(self, x):
            outputs = self.base_model(pixel_values=x)
            probs = self.softmax(outputs.logits)
            return probs
    
    wrapper = ModelWrapper(model)
    wrapper.eval()
    
    # Create dummy input and trace
    dummy_input = torch.randn(1, 3, height, width)
    print("Tracing model...")
    traced_model = torch.jit.trace(wrapper, dummy_input)
    
    # Convert to Core ML
    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="input",
                shape=(1, 3, height, width),
                scale=1.0 / 255.0
            )
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16
    )
    
    # Save
    output_path = OUTPUT_DIR / "RecyclingNet11.mlpackage"
    print(f"\nSaving to {output_path}...")
    mlmodel.save(str(output_path))
    
    size_mb = output_path.stat().st_size / (1024**2)
    print(f"✅ RecyclingNet11 saved: {size_mb:.1f} MB")

def export_food_detector():
    """Export FoodDetector (EfficientNet-B2) to Core ML."""
    print("\n" + "=" * 70)
    print("EXPORTING FOODDETECTOR (EFFICIENTNET-B2)")
    print("=" * 70)
    
    model_path = find_model_snapshot(EFFICIENTNET_DIR)
    if not model_path:
        print("❌ EfficientNet-B2 model not found")
        return
    
    print(f"Loading from: {model_path}")
    
    # Load model and processor
    model = AutoModelForImageClassification.from_pretrained(str(model_path))
    processor = AutoImageProcessor.from_pretrained(str(model_path))
    
    # Get input size
    if hasattr(processor, 'size'):
        size_dict = processor.size
        if isinstance(size_dict, dict):
            height = size_dict.get('height', 224)
            width = size_dict.get('width', 224)
        else:
            height = width = size_dict
    else:
        height = width = 224
    
    print(f"Input size: {height}x{width}")
    print("Classes: 2 (food / non-food)")
    
    # Create wrapper
    class ModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.softmax = torch.nn.Softmax(dim=1)
        
        def forward(self, x):
            outputs = self.base_model(pixel_values=x)
            probs = self.softmax(outputs.logits)
            return probs
    
    wrapper = ModelWrapper(model)
    wrapper.eval()
    
    # Trace model
    dummy_input = torch.randn(1, 3, height, width)
    print("Tracing model...")
    traced_model = torch.jit.trace(wrapper, dummy_input)
    
    # Convert to Core ML
    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="input",
                shape=(1, 3, height, width),
                scale=1.0 / 255.0
            )
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16
    )
    
    # Save
    output_path = OUTPUT_DIR / "FoodDetector.mlpackage"
    print(f"\nSaving to {output_path}...")
    mlmodel.save(str(output_path))
    
    size_mb = output_path.stat().st_size / (1024**2)
    print(f"✅ FoodDetector saved: {size_mb:.1f} MB")

def export_food_detector_inverted():
    """Export FoodDetector with inverted labels (non-food=0, food=1)."""
    print("\n" + "=" * 70)
    print("EXPORTING FOODDETECTOR_INVERTED")
    print("=" * 70)
    
    model_path = find_model_snapshot(EFFICIENTNET_DIR)
    if not model_path:
        print("❌ EfficientNet-B2 model not found")
        return
    
    print("Loading base model...")
    model = AutoModelForImageClassification.from_pretrained(str(model_path))
    processor = AutoImageProcessor.from_pretrained(str(model_path))
    
    # Get input size
    if hasattr(processor, 'size'):
        size_dict = processor.size
        if isinstance(size_dict, dict):
            height = size_dict.get('height', 224)
            width = size_dict.get('width', 224)
        else:
            height = width = size_dict
    else:
        height = width = 224
    
    print(f"Input size: {height}x{width}")
    print("Classes: 2 (inverted - non-food=0, food=1)")
    
    # Create wrapper with inverted outputs
    class InvertedWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.softmax = torch.nn.Softmax(dim=1)
        
        def forward(self, x):
            outputs = self.base_model(pixel_values=x)
            logits = outputs.logits
            # Invert: swap class 0 and 1
            inverted_logits = logits[:, [1, 0]]
            probs = self.softmax(inverted_logits)
            return probs
    
    wrapper = InvertedWrapper(model)
    wrapper.eval()
    
    # Trace model
    dummy_input = torch.randn(1, 3, height, width)
    print("Tracing model...")
    traced_model = torch.jit.trace(wrapper, dummy_input)
    
    # Convert to Core ML
    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="input",
                shape=(1, 3, height, width),
                scale=1.0 / 255.0
            )
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16
    )
    
    # Save
    output_path = OUTPUT_DIR / "FoodDetector_inverted.mlpackage"
    print(f"\nSaving to {output_path}...")
    mlmodel.save(str(output_path))
    
    size_mb = output_path.stat().st_size / (1024**2)
    print(f"✅ FoodDetector_inverted saved: {size_mb:.1f} MB")

def main():
    """Export all models."""
    print("\n" + "=" * 70)
    print("CORE ML MODEL EXPORT PIPELINE")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        export_recycling_net()
    except Exception as e:
        print(f"⚠️  RecyclingNet11 export failed: {e}")
    
    try:
        export_food_detector()
    except Exception as e:
        print(f"⚠️  FoodDetector export failed: {e}")
    
    try:
        export_food_detector_inverted()
    except Exception as e:
        print(f"⚠️  FoodDetector_inverted export failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ EXPORT PIPELINE COMPLETE")
    print("=" * 70)
    
    # Show results
    models = list(OUTPUT_DIR.glob("*.mlpackage"))
    print(f"\nGenerated {len(models)} Core ML models:")
    for model_path in sorted(models):
        size_mb = model_path.stat().st_size / (1024**2)
        print(f"  ✅ {model_path.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
