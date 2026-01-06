**MPS (Metal) & ANE guidance**

- **MPS (Metal Performance Shaders)**: use this for accelerated training/inference with PyTorch on Apple Silicon (M1/M2/M3/M4). Use `torch.device("mps")` when `torch.backends.mps.is_available()`.
- **ANE (Apple Neural Engine)**: used by Core ML for on‑device inference (iPhone/iPad). You do not train on ANE — convert your model to Core ML and set `computeUnits = .all` to let Core ML dispatch to ANE/GPU/CPU.

Practical tips for training on Mac (MPS):
- Set `pin_memory=False` (MPS does not use pinned memory). The trainer now chooses `pin_memory=False` automatically for MPS.
- Use fewer DataLoader workers for MPS (e.g., `num_workers=0` or `1`). The trainer sets `num_workers=0` for MPS.
- Use batch sizes that fit memory; if you run out of memory, lower batch size.
- If you encounter unsupported ops on MPS, try `torch.backends.mps.is_available()` checks and fall back to CPU for those steps.

Converting to Core ML and running on ANE:
- Convert your final checkpoint to ONNX or a TF SavedModel and then to Core ML using `coremltools`.
- When creating the Core ML model, use `compute_units=ct.converters.MILConverterComputeUnits.ALL` (or set `computeUnits = .all` in Swift) to enable ANE.

Suggested local workflow:
1. Iterative development: PyTorch + MPS on your Mac for quick fine-tuning and validation.
2. Final training: longer runs on a GPU VM or keep using MPS if acceptable.
3. Convert best checkpoint to Core ML and test on-device (ANE) for latency and accuracy.

File references:
- `scripts/train_food_binary_torch.py` — uses MPS when available and tunes DataLoader settings.
- `scripts/train_food_binary.py` — Keras training script (requires TensorFlow environment).
