# Quality & Verification Plan

## Testing Layers
- **Unit Tests**: XCTest covering Core logic (model loader, fusion rules, telemetry sanitization). Run on every commit via CI.
- **Snapshot/UI Tests**: XCUITest suites to validate SwiftUI flows, accessibility labels, VoiceOver hints. Added in Phase 2.
- **Performance Tests**: XCTest performance harness measuring inference latency, capture loop throughput, and energy usage (run nightly in CI with device farm or manual instrumentation).
- **Static Analysis**: SwiftLint (style), SwiftFormat (format), Xcode static analyzer, and Swift compiler warnings treated as errors.
- **Security Checks**: Automated script scanning for file write APIs in production modules; manual quarterly audit of permissions and network usage.

## Tooling
- `swift test` with parallel execution.
- Instruments (Time Profiler, Energy Log) baseline profiles stored under `docs/perf/`.
- Manual smoke tests on target devices before every release candidate.

## Release Gates
1. All automated tests pass (unit, UI, performance smoke).
2. Linting/formatting clean.
3. Privacy checklist validated (no new data collection paths without approval).
4. Manual regression sign-off on latest supported iOS versions.

## Regression Handling
- Fail forward: revert commits only if critical production regression; otherwise ship hotfix from patch branch.
- Maintain changelog summarizing fixes and mitigations.
- Track classification accuracy drift through held-out validation dataset scored in training pipeline.
