# Agent Session Notes — 2026-01-06

## Session Summary

- Calibrated and fixed FoodDetector Core ML input scaling; resolved inverted food probabilities.
- Added automatic output-format detection for the FoodDetector (`detect_food_prob_fn`) and wired it into `scripts/benchmark_coreml.py`.
- Ran the benchmark with `--runs 50 --per-run` and saved per-run results to `outputs/benchmarks/benchmark_latest_50.txt`.
- Removed the old inverted-results file `outputs/benchmarks/benchmark_latest.txt`.
- Updated the in-repo todo list to add a task to curate close-up waste-type test images under `data_balanced/test_waste/`.

## Next To-Do Steps

- Curate/download a balanced set of close-up, single-object waste images for waste-type testing and place them in `data_balanced/test_waste/` (organized by class).
- Implement `preprocessImage(_:)` in `Sources/RecycleMVPKit/ModelLoader.swift` to match the Core ML model preprocessing (resize, channel order, mean/std normalization).
- Run quantization experiments (FP16 and 8-bit) for both `FoodDetector` and `RecyclingNet11`, convert, and re-run latency/accuracy benchmarks.
- Create an on-device profiling run (Xcode + device, ANE profiling) to measure real-world latencies and power.
- Commit the benchmark fixes and new notes; add a short README for how to reproduce benchmarks in `coreml_env`.

## Files Created/Modified in This Agent Session

- Modified: `scripts/benchmark_coreml.py` — added `detect_food_prob_fn()` and calibration wiring.
- Created: `outputs/benchmarks/benchmark_latest_50.txt` — per-run benchmark output (50 runs, saved).
- Deleted: `outputs/benchmarks/benchmark_latest.txt` — old benchmark file containing inverted results (removed).
- Created: `agent-session-notes.md` — this file.

(If you want, I can commit these changes and the new notes to a branch.)
