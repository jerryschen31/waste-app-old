# Recycle MVP

This repository hosts the iOS waste-stream identification MVP. The goal is to deliver a privacy-preserving app that classifies household waste items into recycling, compost, or trash while running inference on device. Phase 0 establishes guardrails, documentation, and automation to keep future development safe and predictable.

## Repository Layout

- `docs/` – specifications, privacy model, and engineering references.
- `Sources/` – Swift package sources for shareable logic.
- `Tests/` – XCTest bundles validating shared logic.
- `scripts/` – developer tooling and CI helpers.
- `.github/workflows/` – continuous integration configuration.

## Getting Started

```bash
make setup   # optional helper: installs tooling when available
make test    # runs unit tests via Swift Package Manager
```

See `docs/requirements.md` for functional scope and `docs/threat-model.md` for the security baselines guiding the project.
