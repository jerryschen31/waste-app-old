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
    # Training script used 0=food, 1=not_food (based on alphabetical sort of folders 'food', 'not_food')
    # Core ML ClassifierConfig matches index of output vector to label list.
    # Our output vector from SoftmaxWrapper is [p_not_food, p_food] (from previous logic)
    # Wait, SoftmaxWrapper logic: 
    #   p_food = sigmoid(logits)
    #   p_not_food = 1.0 - p_food
    #   return torch.cat([p_not_food, p_food], dim=1)
    # So Index 0 is "Not Food", Index 1 is "Food".
    #
    # BUT, if the model learned 0=Food (negative logic for p_food?)
    # train_food_binary_torch.py used 0=Food, 1=Not Food.
    # The model outputs a single logit.
    # BCEWithLogitsLoss(pos_weight=...) was used.
    # pos_weight is for the positive class (target=1).
    # Target 1 was "Not Food".
    # So the model trained to predict "Not Food" as 1 (high logit) and "Food" as 0 (low logit).
    #
    # So:
    # High Logit -> High p_food (in my wrapper below) -> this is wrong variable naming
    # wrapper.p_food = sigmoid(logits) -> This gives Probability of Class 1 (Not Food).
    # wrapper.p_not_food = 1 - p_food -> This gives Probability of Class 0 (Food).
    # 
    # Wrapper returns: [p_not_food, p_food] -> [Prob(Food), Prob(Not Food)]
    # So Index 0 is Food. Index 1 is Not Food.
    #
    # Previous code: class_labels = ["not_food", "food"] -> Label 0="not_food", Label 1="food".
    # Previous Result:
    #   Model sees Food -> Low Logit -> Low Sigmoid (Prob Class 1 low) -> High Index 0.
    #   Index 0 labeled "not_food".
    #   Food -> Classified as "not_food".
    #   This explains why Food was NOT detected as Food.
    #
    #   Model sees Non-Food -> High Logit -> High Sigmoid (Prob Class 1 high) -> High Index 1.
    #   Index 1 labeled "food".
    #   Non-Food -> Classified as "food".
    #   This explains why Recycling (Non-Food) was classified as Compost (Food).
    #
    # CORRECTION:
    # Index 0 is Food. Index 1 is Not Food.
    class_labels = ["food", "not_food"] 
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
