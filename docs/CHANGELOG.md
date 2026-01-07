# CHANGELOG — Recent agent session(s)

## 2026-01-06 / 2026-01-07 — Phase 1 work (session summary)

- Introduced a two-stage inference cascade:
  - `FoodDetector` (binary) runs first per detected object. If the item is classified as food, the app returns `Compost` and skips the multi-class model.
  - Otherwise `RecyclingNet11` runs (11 classes) and its outputs are aggregated into 5 disposal categories: `Trash`, `Recycle`, `Compost`, `Biological Waste`, `E-waste`.

- Files added/modified in this session:
  - `scripts/benchmark_coreml.py` — added automatic FoodDetector output calibration and per-run printing.
  - `scripts/convert_to_coreml.py` — conversion paths extended for PyTorch checkpoint → Core ML.
  - `Sources/RecycleMVPKit/ModelLoader.swift` — added 11→5 aggregation logic (TODO: implement `preprocessImage`).
  - `docs/*.md` — updated architecture, model-comparison, model-pipeline, and requirements to document the cascade, measured latencies, and updated requirements.
  - `outputs/benchmarks/benchmark_latest_50.txt` — saved 50-run corrected per-run benchmark results.
  - `agent-session-notes.md` and `new agent-session-notes-2.md` — session notes and multi-day summary.

- Measured benchmark baselines (development machine; use as relative guidance):
  - `FoodDetector` (binary): mean ~1.19 ms, median ~1.16 ms, p95 ~1.33 ms (50-run)
  - `RecyclingNet11` (11-way): mean ~4.59 ms, median ~4.58 ms, p95 ~4.69 ms (50-run)
  - Combined (naive sum): ~6 ms on the Apple M-series machine used for conversion/benchmarking.

- Notes / Next actions (short):
  - Curate a close-up, single-object waste-type test set under `data_balanced/test_waste/` for more representative validation.
  - Implement `preprocessImage(_:)` in `Sources/RecycleMVPKit/ModelLoader.swift` to match Core ML preprocessing (resize, channel order, normalization).
  - Run quantization experiments (FP16 / int8) and re-benchmark on actual target devices (iPhone + ANE profiling).

---

(End of recent session notes.)
