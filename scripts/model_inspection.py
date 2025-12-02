#!/usr/bin/env python3
"""
Inspect the Recycling-Net-11 model architecture and I/O specs.
Helps understand what conversion will need to handle.

Usage:
    python3 scripts/model_inspection.py
"""

import json
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

MODEL_PATH = "./models/recycling-net-11/models--prithivMLmods--Recycling-Net-11/snapshots/6205d424ed3b7feb16faa4e599f986663a1dbbb7"

def inspect_model():
    """Load and inspect model architecture."""
    
    print("=" * 70)
    print("RECYCLING-NET-11 MODEL INSPECTION")
    print("=" * 70)
    
    # Load configuration
    config_path = Path(MODEL_PATH) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    print("\n1. MODEL ARCHITECTURE")
    print(f"   Architecture Type: {config.get('architectures', ['unknown'])[0]}")
    print(f"   Base Model: {config.get('model_type', 'unknown')}")
    print(f"   Hidden Dimensions: {config.get('hidden_size', 'N/A')}")
    print(f"   Number of Layers: {config.get('num_hidden_layers', 'N/A')}")
    
    # Load image processor
    processor_path = Path(MODEL_PATH) / "preprocessor_config.json"
    with open(processor_path) as f:
        processor_config = json.load(f)
    
    print("\n2. INPUT PREPROCESSING")
    print(f"   Input Image Size: {processor_config.get('size', {}).get('height', 'N/A')}x{processor_config.get('size', {}).get('width', 'N/A')}")
    print(f"   Normalization Mean: {processor_config.get('image_mean', [])}")
    print(f"   Normalization Std: {processor_config.get('image_std', [])}")
    print(f"   Do Resize: {processor_config.get('do_resize', True)}")
    print(f"   Do Normalize: {processor_config.get('do_normalize', True)}")
    
    # Load model and count parameters
    print("\n3. LOADING MODEL (this may take a moment)...")
    try:
        model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        print(f"   Model Size (float32): ~{total_params * 4 / (1024**2):.1f} MB")
        
        # Get number of output classes
        num_labels = config.get('num_labels', 'unknown')
        print(f"\n4. OUTPUT")
        print(f"   Number of Classes: {num_labels}")
        
        # List class labels if available
        if hasattr(model.config, 'id2label'):
            print(f"   Classes:")
            for idx, label in sorted(model.config.id2label.items(), key=lambda x: int(x[0])):
                print(f"      {idx}: {label}")
        
        # Test inference shape
        print(f"\n5. TEST INFERENCE SHAPE")
        processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        
        # Create dummy input
        from PIL import Image
        dummy_image = Image.new('RGB', (processor_config['size']['height'], processor_config['size']['width']))
        inputs = processor(dummy_image, return_tensors="pt")
        
        print(f"   Input tensor shape: {inputs['pixel_values'].shape}")
        print(f"   Input tensor dtype: {inputs['pixel_values'].dtype}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"   Output shape: {outputs.logits.shape}")
        print(f"   Output dtype: {outputs.logits.dtype}")
        
        print("\n✓ Model inspection complete!")
        
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_model()
