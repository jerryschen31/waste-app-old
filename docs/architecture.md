# Architecture Overview (Phase 0 Snapshot)

```
+-------------------------+      +---------------------+
| SwiftUI App Shell       |      | Safety Net          |
| - Navigation            |<---->| - Crash handling    |
| - App state             |      | - Watchdogs         |
+------------+------------+      +----------+----------+
             |                              ^
             v                              |
+------------+------------+      +----------+----------+
| CameraCaptureKit        |----->| Telemetry Broker    |
| - AVCaptureSession      |      | - Opt-in metrics    |
| - Lifecycle mgmt        |      | - Secure storage    |
+------------+------------+      +----------+----------+
             |                              ^
             v                              |
+------------+------------+      +----------+----------+
| Image Preprocessor      |----->| Model Lifecycle Mgr |
| - Vision pipelines      |      | - Bundled/remote    |
| - Buffer validation     |      | - Integrity checks  |
+------------+------------+      +----------+----------+
             |                              ^
             v                              |
+------------+------------+      +----------+----------+
| Inference Engine        |----->| Recommendation Svc  |
| - Core ML runtime       |      | - Policy rules      |
| - ANE scheduling        |      | - Explanations      |
+------------+------------+      +----------+----------+
             |                              ^
             v                              |
+------------+------------+      +----------+----------+
| Signal Fusion           |------+ Waste Stream Output |
| - Classifier confidence |      | - UI presentation   |
| - Text/barcode cues     |      | - Manual override   |
+-------------------------+      +---------------------+
```

## Module Responsibilities
- **CameraCaptureKit** abstracts camera access and ensures frames are delivered safely to the pipeline without leaking resources.
- **Image Preprocessor** performs deterministic resizing/normalization so inference receives stable tensors and includes guard clauses for malformed buffers.
- **Inference Engine** wraps Core ML with configuration for ANE acceleration and timeouts to avoid UI blocking.
- **Signal Fusion** combines multiple signals (model, heuristics, text) and manages confidence thresholds.
- **Recommendation Service** uses signed policy data to produce disposal guidance and user-facing explanations.
- **Model Lifecycle Manager** validates and swaps models via signed manifests; interacts with Safety Net for fallback.
- **Telemetry Broker** limits collection to opt-in aggregates, stored encrypted and uploaded opportunistically.
- **Safety Net** centralizes error handling, crash reporting hooks, and watchdog timers.

The Swift Package in this repo captures logic for Inference Engine, Signal Fusion, Recommendation Service, and supporting utilities. SwiftUI shells and AV capture layers will be added in Phase 1.
