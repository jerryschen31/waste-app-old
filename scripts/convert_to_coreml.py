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
import torchvision.models as tv_models
import torch.nn as nn

MODEL_PATH = "./models/recycling-net-11/models--prithivMLmods--Recycling-Net-11/snapshots/6205d424ed3b7feb16faa4e599f986663a1dbbb7"
OUTPUT_DIR = "./models/coreml"
PYTORCH_CHECKPOINT = "./outputs/ft_partial_mps/best_model.pth"  # fine-tuned EfficientNet-B2 binary head

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
    
    # Convert ONNX to Core ML (use generic ct.convert entrypoint)
    mlmodel = ct.convert(
        onnx_path,
        source='onnx',
        minimum_deployment_target=ct.target.iOS16,  # iOS 16+
        compute_units=ct.ComputeUnit.ALL,  # Use CPU, GPU, and ANE
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


def pytorch_checkpoint_to_coreml(checkpoint_path: str, height: int = 224, width: int = 224):
    """Convert a PyTorch EfficientNet-B2 checkpoint (binary head) to Core ML.

    This function builds the same torchvision EfficientNet-B2 architecture used
    for training, loads the `state_dict`, wraps the model with preprocessing
    (resize/normalize), traces it, and converts to a `.mlpackage`.
    """
    print("\n" + "=" * 70)
    print("STEP: PYTORCH CHECKPOINT → CORE ML (EfficientNet-B2 Food Detector)")
    print("=" * 70)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build model architecture (same as training)
    print("Building EfficientNet-B2 architecture...")
    model = tv_models.efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features if hasattr(model, 'classifier') else model.fc.in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, 1))

    # Load state dict
    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(str(checkpoint_path), map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    # Wrap model to include preprocessing: expects input in range [0,255], HxW
    class WrappedModel(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        def forward(self, x):
            # x expected as float32 tensor with values in [0,255]
            x = x / 255.0
            x = (x - self.mean) / self.std
            logits = self.base(x)
            probs = torch.sigmoid(logits)
            return probs

    wrapped = WrappedModel(model)

    # Dummy input: NCHW, values in [0,255]
    dummy = torch.rand(1, 3, height, width) * 255.0

    print("Tracing model with torch.jit.trace...")
    traced = torch.jit.trace(wrapped, dummy)

    # Convert traced module to Core ML using tensor input (we keep raw float input)
    print("Converting traced model to Core ML (.mlpackage)...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=dummy.shape)],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
    )

    # Metadata
    mlmodel.author = "Recycle MVP"
    mlmodel.short_description = "Food / not-food binary detector (EfficientNet-B2 fine-tuned)"
    mlmodel.version = "1.0.0"
    # Set input/output descriptions using actual feature names from the spec
    try:
        spec = mlmodel.get_spec()
        if spec.description.input:
            in_name = spec.description.input[0].name
            mlmodel.input_description[in_name] = f"Input image tensor (1,3,{height},{width}) with values in [0,255]"
        if spec.description.output:
            out_name = spec.description.output[0].name
            mlmodel.output_description[out_name] = "Probability of being food (0..1)"
    except Exception:
        pass

    save_path = out_path / "FoodDetector.mlpackage"
    print(f"Saving Core ML model to: {save_path}")
    mlmodel.save(str(save_path))
    print(f"✓ Saved FoodDetector Core ML model. Size: {save_path.stat().st_size / (1024**2):.2f} MB")

    return str(save_path)


def pytorch_hf_to_coreml(model_path: str):
    """Convert the Hugging Face Recycling-Net PyTorch snapshot directly to Core ML.

    This traces a wrapper that normalizes inputs (expects [0,1] float tensor)
    and returns softmax logits.
    """
    print("\n" + "=" * 70)
    print("STEP: HUGGINGFACE PYTORCH SNAPSHOT → CORE ML (Recycling-Net-11)")
    print("=" * 70)

    mp = Path(model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model path not found: {mp}")

    print("Loading model from Hugging Face snapshot...")
    model = AutoModelForImageClassification.from_pretrained(model_path)

    # Attempt to read input size and normalization from stored config
    config_path = Path(model_path) / "preprocessor_config.json"
    if config_path.exists():
        with open(config_path) as f:
            proc = json.load(f)
        h = proc.get('size', {}).get('height', 224)
        w = proc.get('size', {}).get('width', 224)
        mean = proc.get('image_mean', [0.485, 0.456, 0.406])
        std = proc.get('image_std', [0.229, 0.224, 0.225])
    else:
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    print(f"Input size: {h}x{w}; mean={mean}; std={std}")

    class WrappedHF(nn.Module):
        def __init__(self, base, mean, std):
            super().__init__()
            self.base = base
            mean_t = torch.tensor(mean).view(1, 3, 1, 1)
            std_t = torch.tensor(std).view(1, 3, 1, 1)
            self.register_buffer('mean', mean_t)
            self.register_buffer('std', std_t)

        def forward(self, x):
            # Expect x in [0,1]
            x = (x - self.mean) / self.std
            logits = self.base(x).logits if hasattr(self.base, 'forward') else self.base(x)
            return logits

    wrapped = WrappedHF(model, mean, std)
    wrapped.eval()

    dummy = torch.rand(1, 3, h, w)
    print("Tracing HF model with torch.jit.trace...")
    traced = torch.jit.trace(wrapped, dummy)

    print("Converting traced HF model to Core ML (.mlpackage)...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name='input', shape=dummy.shape)],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
    )

    mlmodel.author = "Recycle MVP"
    mlmodel.short_description = "Waste item classification (11 categories) - RecyclingNet11"
    mlmodel.version = "1.0.0"
    # Set input/output descriptions using actual feature names from the spec
    try:
        spec = mlmodel.get_spec()
        if spec.description.input:
            in_name = spec.description.input[0].name
            mlmodel.input_description[in_name] = f"Input image tensor (1,3,{h},{w}) in [0,1]"
        if spec.description.output:
            out_name = spec.description.output[0].name
            mlmodel.output_description[out_name] = "Logits for 11 classes"
    except Exception:
        pass

    save_path = Path(OUTPUT_DIR) / "RecyclingNet11.mlpackage"
    print(f"Saving Core ML model to: {save_path}")
    mlmodel.save(str(save_path))
    print(f"✓ Saved RecyclingNet11 Core ML model. Size: {save_path.stat().st_size / (1024**2):.2f} MB")
    return str(save_path)

def main():
    """Main conversion pipeline"""
    
    print("\n" + "=" * 70)
    print("RECYCLING-NET-11 CONVERSION PIPELINE")
    print("PyTorch → ONNX → Core ML")
    print("=" * 70)
    
    try:
        # Step 1: Convert HF PyTorch snapshot directly to Core ML (avoid ONNX path)
        coreml_path = pytorch_hf_to_coreml(MODEL_PATH)

        print("\n" + "=" * 70)
        print("✓ Recycling-Net-11 conversion complete")
        print("=" * 70)
        print(f"\nCore ML Recycling model ready at: {coreml_path}")

        # If a fine-tuned PyTorch checkpoint exists, convert it too
        ckpt = Path(PYTORCH_CHECKPOINT)
        if ckpt.exists():
            print("\nFound fine-tuned PyTorch checkpoint; converting FoodDetector...")
            fd_path = pytorch_checkpoint_to_coreml(str(ckpt))
            print(f"\nCore ML Food Detector ready at: {fd_path}")
        else:
            print("\nNo fine-tuned PyTorch checkpoint found; skipping FoodDetector conversion.")

        print("\nNext steps:")
        print("1. Integrate .mlpackage files into Xcode project")
        print("2. Add to Sources/RecycleMVPKit as model resource")
        print("3. Implement inference wrapper in Inference Engine module")
        print("4. Run latency tests on target device")
        
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
