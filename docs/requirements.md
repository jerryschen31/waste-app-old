# Recycle MVP Requirements

## 1. Product Scope
- Provide iOS users with a fast, offline-capable assistant for classifying household waste items into **recycling**, **compost**, or **landfill/trash** streams.
- Operate primarily on-device, leveraging Core ML and Apple Neural Engine (ANE) for inference.
- Offer explainable recommendations (e.g., "Aluminum can → recycling") and opportunity for manual override.
- Support iPhone devices running iOS 16+; prioritize iPhone 12 and newer hardware for performance baselines.

## 2. User Stories
1. **Scan Item**: As a user, I can capture a photo of a waste item and receive a recommended disposal stream with confidence score.
2. **Manual Override**: As a user, I can correct the recommendation, which feeds future model improvement telemetry (if opted-in).
3. **Offline Use**: As a user, I can classify items without network access.
4. **Explainability**: As a user, I understand why an item was classified via concise textual explanation or iconography.
5. **History (Optional Post-MVP)**: As a user, I can review recent classifications (without storing images).
6. **Accessibility**: As a user, I can operate the app with VoiceOver and large text modes.

## 3. Functional Requirements
- **For Capture**: Camera pipeline must deliver still-image inference within 600ms median latency on target hardware.
- **For Inference**: Core ML model packaged with the app; model update mechanism may download signed bundles.
- **Confidence Handling**: Low-confidence predictions (< configurable threshold) must prompt user confirmation before finalizing.
- **Telemetry**: Collect only opt-in, anonymized aggregates (counts, corrections). Never store or transmit raw images.
- **Error States**: Display actionable messages for lack of camera permission, inference failure, or outdated model.

## 4. Non-Functional Requirements
- **Performance**: Maintain 30fps preview; inference must not exceed 1.5s in worst case. Energy impact measured via Xcode Instruments must stay within "Low" classification for 3-minute session.
- **Reliability**: 99% crash-free sessions target. Automated tests required for every module change.
- **Security & Privacy**: Only metadata stored locally. Use secure storage for tokens. Enforce HTTPS with certificate pinning for any network calls.
- **Maintainability**: Code structured in modules with clear boundaries. Document architecture and testing strategy. Enforce linting and formatting via CI.
- **Accessibility**: Conform to WCAG 2.1 AA for color contrast and dynamic type.

## 5. Out of Scope (MVP)
- Android or web clients.
- Cloud-first inference.
- User accounts or social sharing features.
- Persistent image storage.

## 6. Success Metrics
- ≥90% accuracy on curated validation dataset covering top 50 waste items.
- Median inference latency <600ms per item on iPhone 13.
- ≤1% of sessions reporting crash or critical failure.
- ≥80% of beta users report trust in recommendation explanations.

## 7. Open Questions
- What geographic locales do we target for waste rules variations? (Impacts policy JSON.)
- How frequently should model updates be pushed? Determine cadence after telemetry review.
- Which dataset licensing constraints apply for training imagery? Resolve before Phase 2.
