# Privacy & Threat Model

## Objectives
- Preserve user privacy by preventing waste-item photographs from leaving the device without explicit consent.
- Secure optional telemetry and model update channels against tampering and data leakage.
- Ensure adversaries cannot poison or replace on-device models.

## Assets
- **User imagery**: transient camera frames and processed tensors.
- **Classification results**: disposal stream labels, confidence scores, correction inputs.
- **Policy configuration**: signed JSON mapping categories to waste streams.
- **Telemetry aggregates**: anonymized counters.
- **Model binaries**: bundled and remotely distributed `.mlmodelc` files.

## Adversaries & Threats
- **External network attacker**: attempts MITM against telemetry or model update endpoints.
- **Malicious local app**: tries to access stored data or intercept inter-process communication.
- **Tampering adversary**: attempts to replace model/policy files with compromised versions.
- **Curious insider**: misuses telemetry storage or logs to infer personal information.

## Mitigations
- **On-device inference**: images kept in memory only; enforce `CVPixelBuffer` zeroing and no disk persistence.
- **Secure storage**: policies and telemetry keys stored via `Keychain`; local aggregates encrypted with per-installation key.
- **Network security**: HTTPS with certificate pinning; signed URLs with expirations; telemetry payloads contain no PII.
- **Integrity checks**: verify SHA-256 and signed manifests before activating downloaded models/policies.
- **Permissions**: request camera access only when needed; provide manual fallback for denied permission to avoid dark patterns.
- **Logging discipline**: disable verbose logging in production builds; sanitize error reports.
- **CI enforcement**: static analysis ensures no inadvertent file writes of raw images; unit tests cover privacy-sensitive flows.

## Residual Risks
- Physical device compromise (jailbreak) exposes memory; mitigated by standard iOS protections, but out of scope.
- User sharing screenshots bypasses privacy controls; notify users about responsible usage.
- Future features requiring cloud inference would introduce new threat vectors; re-evaluate before implementation.
