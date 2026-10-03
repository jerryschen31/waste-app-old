#!/usr/bin/env python3
"""Export models to ONNX without requiring coremltools.

1) Exports the Hugging Face Recycling-Net-11 snapshot to ONNX.
2) Exports the fine-tuned EfficientNet-B2 checkpoint (binary head) to ONNX.

This is a fallback when `coremltools` is not available in the environment.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as tv_models
from transformers import AutoModelForImageClassification
import json


MODEL_PATH = "./models/recycling-net-11/models--prithivMLmods--Recycling-Net-11/snapshots/6205d424ed3b7feb16faa4e599f986663a1dbbb7"
CHECKPOINT = "./outputs/ft_partial_mps/best_model.pth"
OUTDIR = Path("./models/coreml")
OUTDIR.mkdir(parents=True, exist_ok=True)


def export_recycling_net():
    print("Exporting Recycling-Net-11 (HF snapshot) to ONNX...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
    # Read processor config for size
    config_path = Path(MODEL_PATH) / "preprocessor_config.json"
    with open(config_path) as f:
        pc = json.load(f)
    h = pc['size']['height']
    w = pc['size']['width']

    dummy = torch.randn(1, 3, h, w)
    onnx_path = OUTDIR / "recycling_net_11.onnx"
    print(f" - input size: {h}x{w}; saving to {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={'pixel_values': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
        opset_version=14,
    )
    print("Done.")
    return onnx_path


def export_food_detector():
    ckpt = Path(CHECKPOINT)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}. Skipping FoodDetector export.")
        return None

    print("Exporting fine-tuned EfficientNet-B2 (FoodDetector) to ONNX...")
    model = tv_models.efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features if hasattr(model, 'classifier') else model.fc.in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, 1))

    state = torch.load(str(ckpt), map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    # Wrap to include normalization and sigmoid -> output probability
    class Wrapped(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        def forward(self, x):
            # Expect input in [0,1] (as produced by ToTensor())
            x = (x - self.mean) / self.std
            logits = self.base(x)
            probs = torch.sigmoid(logits)
            return probs

    wrapped = Wrapped(model)
    dummy = torch.rand(1, 3, 224, 224)
    onnx_path = OUTDIR / "FoodDetector.onnx"
    torch.onnx.export(
        wrapped,
        dummy,
        str(onnx_path),
        input_names=['input'],
        output_names=['prob'],
        dynamic_axes={'input': {0: 'batch_size'}, 'prob': {0: 'batch_size'}},
        opset_version=14,
    )
    print(f"Saved FoodDetector ONNX to {onnx_path}")
    return onnx_path


def main():
    r = export_recycling_net()
    f = export_food_detector()
    print('\nExport summary:')
    if r:
        print(f' - Recycling ONNX: {r} ({r.stat().st_size / (1024**2):.2f} MB)')
    if f:
        print(f' - FoodDetector ONNX: {f} ({f.stat().st_size / (1024**2):.2f} MB)')


if __name__ == '__main__':
    main()
