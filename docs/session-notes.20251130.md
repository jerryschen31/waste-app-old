# Session Notes – November 30, 2025

## Objective
Get up to speed on the current state of the waste-app codebase and understand technical/architectural decisions made so far.

---

## Current State of Waste App (Recycle MVP)

### Project Goal
An iOS app that classifies household waste items into **recycling**, **compost**, or **trash** using **on-device neural inference** via Core ML and Apple's Neural Engine (ANE). All processing happens locally—images never leave the device.

### Phase 0 Status (Current)
The project is in the **foundation phase**, focused on establishing guardrails, documentation, and core abstractions before building the full app UI. The Swift Package contains reusable logic; SwiftUI shell and camera integration are planned for Phase 1.

---

## Key Architectural Decisions

| Component | Purpose | Implementation Detail |
|-----------|---------|----------------------|
| **CameraCaptureKit** | Frame delivery | AVCaptureSession with lifecycle safety and buffer validation |
| **Image Preprocessor** | Tensor preparation | Vision framework pipelines for deterministic resizing/normalization |
| **Inference Engine** | Model execution | Core ML runtime with ANE acceleration and 1.5s timeout |
| **Signal Fusion** | Multi-signal combination | Merges model confidence with text/barcode heuristics |
| **Recommendation Service** | Disposal guidance | Uses signed policy JSON and applies confidence thresholds |
| **Model Lifecycle Manager** | Model updates | Validates signed manifests before swapping bundled/remote models |
| **Telemetry Broker** | Privacy-first metrics | Opt-in aggregates only, encrypted locally, no PII or images |
| **Safety Net** | Error handling | Centralized crash reporting, watchdog timers, fallback flows |

---

## Privacy & Security Stance

- **Zero image persistence**: Frames stay in memory only; enforced pixel buffer zeroing
- **Signed integrity checks**: SHA-256 and manifest validation for all models/policies
- **Network hardening**: HTTPS with certificate pinning for telemetry and model updates
- **Secure storage**: Keychain for sensitive data; encrypted local aggregates
- **No dark patterns**: Manual camera fallback if permission denied; clear user consent

---

## Performance Baselines

- **Target latency**: <600ms median inference on iPhone 13+
- **Preview FPS**: 30fps maintained during capture
- **Energy**: Must stay within "Low" classification (Xcode Instruments)
- **Reliability**: 99% crash-free sessions target

---

## Current Code Structure

- **1 implementation file**: `Configuration.swift` — defines confidence threshold (0.7) and inference timeout (1.5s)
- **Swift Package**: Targets iOS 16+ and macOS 13+; no external dependencies yet
- **Test infrastructure**: XCTest suite ready for Phase 1 logic
- **CI/CD**: Makefile with lint, format, test, and verify targets

### What's Not Yet Built
- Camera UI and SwiftUI screens
- Actual Core ML model integration (model files not yet added)
- Network endpoints for telemetry/model updates
- Full test coverage (pending implementation of other modules)

---

## Development Workflow

```bash
make setup    # Install Python tooling (coremltools for model conversion)
make test     # Run XCTest suite
make lint     # Check Swift style (swiftlint)
make format   # Auto-format code (swiftformat)
make verify   # Run CoreML verification script
```

---

## Model Selection Decision

### Question: Has an image-recognition model been chosen yet?

**Answer**: No—Phase 0 has focused on architecture and guardrails. Model selection is a Phase 1+ task.

### Recommended Model: Recycling-Net-11

After searching Hugging Face, the best open-source option for your use case is:

**`prithivMLmods/Recycling-Net-11`** ⭐

- Fine-tuned from **Google SigLIP2** (efficient vision-language model)
- **11 waste categories** (recycling-focused)
- Apache 2.0 license – suitable for open-source projects
- Trained on the `viola77data/recycling-dataset`
- Lightweight architecture ideal for on-device (ANE) inference
- Good conversion path to Core ML format via `coremltools`

### Alternative Options

1. **`prithivMLmods/Trash-Net`** (6-class variant)
   - Same architecture (SigLIP2-based)
   - Categories: cardboard, glass, metal, paper, plastic, trash
   - Simpler if you want to start with fewer categories

2. **MobileNetV2 variants** (e.g., `akmalia31/trash-classification-cnn-mobilnetv2`)
   - Based on the TrashNet dataset
   - Specifically optimized for mobile inference

### Supporting Datasets

- **`garythung/trashnet`** – Original widely-used trash classification dataset (6 classes)
- **`omasteam/waste-garbage-management-dataset`** – 10-class garbage dataset (MIT license)
- **`viola77data/recycling-dataset`** – Used to train Recycling-Net-11

### Why Recycling-Net-11 for Your Project

- ✅ SigLIP2 is lightweight and efficient
- ✅ 11 categories aligns well with detailed waste classification
- ✅ Easy to convert to Core ML format (`coremltools` already in Makefile)
- ✅ ANE-friendly architecture
- ✅ Apache 2.0 license compatible with your project
- ✅ Aligns with your ≥90% accuracy requirement

---

## Next Steps for Model Integration

1. **Download Recycling-Net-11** from Hugging Face
2. **Convert to Core ML** using `coremltools`:
   - Create a Python script to load the HF model and convert to `.mlmodelc`
   - Target ANE acceleration explicitly
3. **Validate Core ML conversion** with the `verify-coreml.sh` script
4. **Test inference latency** on target devices (iPhone 12+)
5. **Create test dataset** for accuracy validation (need ≥90% on top 50 waste items)
6. **Integrate with Inference Engine** module (planned for Phase 1)

---

## Key Requirements Checklist

From `docs/requirements.md`:

- [ ] Model accuracy: ≥90% on curated validation dataset (top 50 waste items)
- [ ] Inference latency: <600ms median on iPhone 13
- [ ] Crash-free: ≤1% of sessions reporting critical failure
- [ ] Energy: ≤"Low" classification per Xcode Instruments
- [ ] Offline capability: All inference on-device
- [ ] iOS 16+ support
- [ ] VoiceOver/accessibility compliance

---

## Resources & Documentation

- **Architecture**: `docs/architecture.md`
- **Requirements**: `docs/requirements.md`
- **Privacy & Threat Model**: `docs/threat-model.md`
- **Quality Plan**: `docs/quality-plan.md`
- **Swift Package**: `Package.swift` (iOS 16+, macOS 13+)
- **Makefile**: Common development tasks

---

## Open Questions for Next Session

1. Which of the top 50 waste items do you want to prioritize for your validation dataset?
2. Do you want to start with Recycling-Net-11 (11 classes) or Trash-Net (6 classes)?
3. Should we create a model conversion + validation pipeline as part of Phase 1?
4. Do you need geographic variations in waste rules (impacts policy JSON structure)?

---

## Files to Review on Return

- `docs/requirements.md` – Full functional scope and success metrics
- `docs/architecture.md` – Detailed module responsibilities
- `docs/quality-plan.md` – Testing strategy and release gates
- `docs/threat-model.md` – Privacy and security baseline

---

**Session Date**: November 30, 2025  
**Workspace**: `/Users/jerry/gh/waste-app`  
**Branch**: master
