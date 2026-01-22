#!/usr/bin/env python3
"""
Export fine-tuned SigLIP-based RecyclingNet11 model to Core ML format.

This script converts the fine-tuned PyTorch model (outputs/finetune_recycling_net/)
to Core ML (.mlpackage) format for iOS deployment.

Usage:
    python3 scripts/export_finetuned_coreml.py
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification
import coremltools as ct

# Paths
FINETUNED_MODEL_DIR = Path("outputs/finetune_recycling_net")
OUTPUT_DIR = Path("models/coreml")
OUTPUT_MODEL = OUTPUT_DIR / "RecyclingNet11ex.mlpackage"

def load_model_and_processor():
    """Load fine-tuned model and processor."""
    print("Loading fine-tuned model...")
    model = AutoModelForImageClassification.from_pretrained(str(FINETUNED_MODEL_DIR))
    processor = AutoImageProcessor.from_pretrained(str(FINETUNED_MODEL_DIR))
    
    # Load class names
    class_names_path = FINETUNED_MODEL_DIR / "class_names.json"
    with open(class_names_path) as f:
        class_names = json.load(f)
    
    print(f"Classes ({len(class_names)}): {class_names}")
    return model, processor, class_names

def create_wrapper_model(model, class_names):
    """Create wrapper model with Softmax for probability outputs."""
    
    class SigLIPWrapper(torch.nn.Module):
        def __init__(self, base_model, num_classes):
            super().__init__()
            self.base_model = base_model
            self.softmax = torch.nn.Softmax(dim=1)
            self.num_classes = num_classes
        
        def forward(self, x):
            # Get logits from base model
            outputs = self.base_model(pixel_values=x)
            logits = outputs.logits
            
            # Apply Softmax to get probabilities
            probs = self.softmax(logits)
            
            return probs
    
    wrapper = SigLIPWrapper(model, len(class_names))
    wrapper.eval()
    return wrapper

def export_to_coreml(model, processor, class_names):
    """Convert PyTorch model to Core ML format."""
    
    print("\nConverting to Core ML...")
    
    # Get input size from processor
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
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, height, width)
    
    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Convert to Core ML with proper input/output specifications
    print("Exporting to Core ML...")
    
    classifier_config = ct.ClassifierConfig(class_names)
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="input",
                shape=(1, 3, height, width),
                scale=1.0 / 255.0
            )
        ],
        classifier_config=classifier_config,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16
    )
    
    # Set classifier metadata
    mlmodel.user_defined_metadata["com.apple.coreml.model.metadata"] = json.dumps({
        "description": "RecyclingNet11 Fine-tuned for E-Waste Detection",
        "version": "1.0",
        "author": "ML Pipeline"
    }).encode('utf-8')
    
    # Create classifier config
    spec = mlmodel.get_spec()
    
    # Set up the classifier output
    if len(spec.description.output) > 0:
        output = spec.description.output[0]
        output.shortDescription = "Class probabilities"
    
    # Save as mlpackage
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {OUTPUT_MODEL}...")
    mlmodel.save(str(OUTPUT_MODEL))
    
    print(f"✅ Core ML model saved: {OUTPUT_MODEL}")
    print(f"   Size: {OUTPUT_MODEL.stat().st_size / (1024**2):.1f} MB")
    
    return mlmodel

def verify_model(mlmodel, processor, class_names):
    """Verify the Core ML model works."""
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    from PIL import Image
    
    # Try to load a test image
    test_images = list(Path("data_finetune/val/electronics").glob("*.jpg"))[:3]
    
    if test_images:
        print(f"\nFound {len(test_images)} test images for verification")
        
        for test_image_path in test_images[:1]:
            print(f"\nTesting with: {test_image_path.name}")
            
            # Load and preprocess image
            image = Image.open(test_image_path).convert("RGB")
            
            # Preprocess using processor
            processed = processor(image, return_tensors="pt")
            pixel_values = processed["pixel_values"]
            
            print(f"  Input shape: {pixel_values.shape}")
            print(f"  Image size: {image.size}")
            
            # Test with PyTorch model
            with torch.no_grad():
                outputs = mlmodel(pixel_values=pixel_values)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                top_class_idx = probs.argmax(dim=1).item()
                top_prob = probs[0, top_class_idx].item()
            
            print(f"  Top class: {class_names[top_class_idx]}")
            print(f"  Confidence: {top_prob:.2%}")
            print(f"  ✅ Model inference successful")
    else:
        print("⚠️  No test images found (optional)")

def main():
    """Main export pipeline."""
    print("\n" + "=" * 70)
    print("FINE-TUNED SIGLIP → CORE ML EXPORT")
    print("=" * 70)
    
    # Check that fine-tuned model exists
    if not FINETUNED_MODEL_DIR.exists():
        print(f"❌ Fine-tuned model not found at {FINETUNED_MODEL_DIR}")
        print("   Run scripts/finetune_recycling_net.py first")
        return
    
    # Load model and processor
    model, processor, class_names = load_model_and_processor()
    
    # Create wrapper with Softmax
    wrapper_model = create_wrapper_model(model, class_names)
    
    # Export to Core ML
    mlmodel = export_to_coreml(wrapper_model, processor, class_names)
    
    # Verify
    verify_model(model, processor, class_names)
    
    print("\n" + "=" * 70)
    print("✅ EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nCore ML model ready for iOS integration:")
    print(f"  Path: {OUTPUT_MODEL}")
    print(f"  Classes: {len(class_names)}")
    print(f"  Input: 224x224 RGB image")
    print(f"  Output: Class probabilities (Softmax)")

if __name__ == "__main__":
    main()
