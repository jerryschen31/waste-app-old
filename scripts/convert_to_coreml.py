#!/usr/bin/env python3
"""
Convert Recycling-Net-11 from PyTorch → ONNX → Core ML.

This creates a .mlpackage file that can be embedded in your iOS app.

Usage:
    python3 scripts/convert_to_coreml.py
"""

import json
from pathlib import Path
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
import coremltools as ct
import onnx

MODEL_PATH = "./models/recycling-net-11/models--prithivMLmods--Recycling-Net-11/snapshots/6205d424ed3b7feb16faa4e599f986663a1dbbb7"
OUTPUT_DIR = "./models/coreml"

def pytorch_to_onnx():
    """Step 1: Convert PyTorch → ONNX"""
    
    print("\n" + "=" * 70)
    print("STEP 1: PYTORCH → ONNX CONVERSION")
    print("=" * 70)
    
    onnx_path = Path(OUTPUT_DIR) / "recycling_net_11.onnx"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load model and processor
    print("Loading model from Hugging Face...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    
    # Get input size from processor config
    config_path = Path(MODEL_PATH) / "preprocessor_config.json"
    with open(config_path) as f:
        processor_config = json.load(f)
    
    height = processor_config['size']['height']
    width = processor_config['size']['width']
    
    print(f"Input image size: {height}x{width}")
    
    # Create dummy input matching expected size
    print("Creating dummy input for tracing...")
    dummy_input = torch.randn(1, 3, height, width)  # (batch=1, channels=3, H, W)
    
    # Export to ONNX
    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14,  # Core ML supports up to opset 14
        verbose=False
    )
    
    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model valid. Size: {onnx_path.stat().st_size / (1024**2):.1f} MB")
    
    return str(onnx_path), height, width

def onnx_to_coreml(onnx_path: str, height: int, width: int):
    """Step 2: Convert ONNX → Core ML"""
    
    print("\n" + "=" * 70)
    print("STEP 2: ONNX → CORE ML CONVERSION")
    print("=" * 70)
    
    coreml_path = Path(OUTPUT_DIR) / "RecyclingNet11.mlpackage"
    
    print(f"Converting {onnx_path} to Core ML...")
    
    # Convert ONNX to Core ML
    mlmodel = ct.converters.onnx.convert(
        onnx_path,
        minimum_deployment_target=ct.target.iOS16,  # iOS 16+
        compute_units=ct.ComputeUnit.ALL  # Use CPU, GPU, and ANE
    )
    
    # Update model metadata
    mlmodel.author = "Recycle MVP"
    mlmodel.license = "Apache 2.0"
    mlmodel.short_description = "Waste item classification (11 categories)"
    mlmodel.version = "1.0.0"
    
    # Set input/output descriptions
    mlmodel.input_description['pixel_values'] = f"Input image ({height}x{width} RGB)"
    mlmodel.output_description['logits'] = "Classification logits (11 classes)"
    
    # Save Core ML model
    print(f"Saving Core ML model to: {coreml_path}")
    mlmodel.save(str(coreml_path))
    
    print(f"✓ Core ML model saved. Size: {coreml_path.stat().st_size / (1024**2):.1f} MB")
    
    return str(coreml_path)

def main():
    """Main conversion pipeline"""
    
    print("\n" + "=" * 70)
    print("RECYCLING-NET-11 CONVERSION PIPELINE")
    print("PyTorch → ONNX → Core ML")
    print("=" * 70)
    
    try:
        # Step 1: PyTorch to ONNX
        onnx_path, height, width = pytorch_to_onnx()
        
        # Step 2: ONNX to Core ML
        coreml_path = onnx_to_coreml(onnx_path, height, width)
        
        print("\n" + "=" * 70)
        print("✓ CONVERSION COMPLETE")
        print("=" * 70)
        print(f"\nCore ML model ready at: {coreml_path}")
        print("\nNext steps:")
        print("1. Integrate RecyclingNet11.mlpackage into Xcode project")
        print("2. Add to Sources/RecycleMVPKit as model resource")
        print("3. Implement inference wrapper in Inference Engine module")
        print("4. Run latency tests on target device")
        
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
