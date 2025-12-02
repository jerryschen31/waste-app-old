# Recycle MVP

This repository hosts the iOS waste-stream identification MVP. The goal is to deliver a privacy-preserving app that classifies household waste items into recycling, compost, or trash while running inference on device. Phase 0 establishes guardrails, documentation, and automation to keep future development safe and predictable.

## Selected Model

**Recycling-Net-11** (Apache 2.0 License)
- **Source**: [prithivMLmods/Recycling-Net-11 on Hugging Face](https://huggingface.co/prithivMLmods/Recycling-Net-11)
- **Architecture**: SigLIP2-based vision-language model
- **Categories**: 11 waste classification categories
- **Status**: Downloaded, ready for Core ML conversion

## Repository Layout

- `docs/` – specifications, privacy model, and engineering references.
- `Sources/` – Swift package sources for shareable logic.
- `Tests/` – XCTest bundles validating shared logic.
- `scripts/` – developer tooling and CI helpers.
- `.github/workflows/` – continuous integration configuration.
- `models/` – ML model artifacts (downloaded and converted models).
- `ml_env/` – Python virtual environment for ML tooling.

## Getting Started

```bash
make setup   # optional helper: installs tooling when available
make test    # runs unit tests via Swift Package Manager
```

See `docs/requirements.md` for functional scope and `docs/threat-model.md` for the security baselines guiding the project.

## Model Conversion Pipeline

See `docs/model-pipeline.md` for step-by-step instructions to convert the Recycling-Net-11 model from PyTorch to Core ML format for on-device inference.
