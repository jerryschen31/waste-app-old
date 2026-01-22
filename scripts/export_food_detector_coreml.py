#!/usr/bin/env python3
"""
Export the PyTorch (Torchvision) trained Food Detector to Core ML.

Loads the binary classifier from outputs/ft_quick_torch/best_model.pth
and exports it as 'FoodDetector.mlpackage'.
"""

import torch
import torch.nn as nn
from torchvision import models
import coremltools as ct
from pathlib import Path

# Configuration
CHECKPOINT_PATH = Path("outputs/ft_quick_torch/best_model.pth")
OUTPUT_PATH = Path("models/coreml/FoodDetector.mlpackage")
INPUT_SIZE = 224

class FoodDetectorWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Model returns shape (Batch, 1)
        logits = self.model(x)
        prob_food = self.sigmoid(logits)
        
        # We need to return a dictionary or something Core ML can use as a classifier
        # But simpler for now: just return the prob array, and we label it manually in Core ML
        return prob_food

def main():
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    
    if not CHECKPOINT_PATH.exists():
        print("❌ Checkpoint not found.")
        return

    # Recreate the model structure used in train_food_binary_torch.py
    # Note: weights=None because we load our own. But safest to init with IMAGENET to match architecture details.
    model = models.efficientnet_b2(weights=None) 
    
    # Replace head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2), 
        nn.Linear(in_features, 1)
    )
    
    # Load weights
    # Using map_location='cpu' to be safe
    state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    # Wrap for probability output
    # But wait, Core ML classifier conversion is cleaner if we output [p_not_food, p_food]
    # Let's make a wrapper that outputs (Batch, 2) so we can use ClassifierConfig easily
    
    class SoftmaxWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            logits = self.base_model(x) # (B, 1)
            p_food = self.sigmoid(logits)
            p_not_food = 1.0 - p_food
            # Concatenate to (B, 2) -> [not_food, food]
            return torch.cat([p_not_food, p_food], dim=1)

    wrapper = SoftmaxWrapper(model)
    wrapper.eval()
    
    # Trace
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    print("Tracing model...")
    traced_model = torch.jit.trace(wrapper, dummy_input)
    
    # Define class labels
    # 0 = not_food, 1 = food
    class_labels = ["not_food", "food"] 
    classifier_config = ct.ClassifierConfig(class_labels)
    
    # ImageNet Normalization
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    # Core ML Scale = 1/(255*std)
    # Core ML Bias = -mean/std
    scale_red   = 1/(255 * 0.229)
    scale_green = 1/(255 * 0.224)
    scale_blue  = 1/(255 * 0.225)
    
    bias_red   = -0.485 / 0.229
    bias_green = -0.456 / 0.224
    bias_blue  = -0.406 / 0.225

    print("Exporting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="input",
                shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
                # Standard ImageNet average scale: 1/(255*0.226) ≈ 0.01735
                scale=0.01735,
                bias=[bias_red, bias_green, bias_blue]
            )
        ],
        classifier_config=classifier_config,
        minimum_deployment_target=ct.target.iOS16
    )
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(OUTPUT_PATH))
    print(f"✅ Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
