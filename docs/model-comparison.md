# Image Classification Model Comparison

## Summary Table

| Model Name | Architecture | Categories | License | Best For | Pros | Cons |
|---|---|---|---|---|---|---|
| **Recycling-Net-11** ⭐ | SigLIP2 (Vision-Language) | 11 (recycling-focused) | Apache 2.0 | **Recommended for MVP** | Efficient, lightweight, domain-optimized, good accuracy-compute trade-off, easy Core ML conversion | Vision-language models may have larger disk footprint; less battle-tested on ANE |
| **Trash-Net** | SigLIP2 (Vision-Language) | 6 (cardboard, glass, metal, paper, plastic, trash) | Apache 2.0 | Simpler classification; legacy devices | Same architecture as Recycling-Net-11, minimal categories | Limited to 6 classes; may lack nuance for recycling subcategories |
| **MobileNetV3 (Large/Small)** | Lightweight CNN | Custom (via transfer learning) | Open source | Transfer learning baseline | Proven mobile backbone, extensive documentation, excellent ANE support | Requires fine-tuning on waste dataset; not pre-trained on waste |
| **EfficientNet-Lite (0/1)** | Mobile-first EfficientNet | Custom (via transfer learning) | Open source | High accuracy-compute trade-off | Better accuracy than MobileNetV3 at similar compute; quantization-friendly | Requires fine-tuning; larger than Lite0 may exceed budget |
| **MobileViT-XXS / MobileViT-XS** | Hybrid Conv+Attention | Custom (via transfer learning) | Open source | High-clutter scenes (multi-item waste bins) | Better generalization on complex scenes; attention helps with occlusion | Larger model; slower inference; overkill for simple item classification |
| **YOLOv8n (Nano)** | Lightweight Object Detector | Custom (via training) | AGPL (check licensing) | Multi-object detection fallback | Detects individual items in scene; enables per-item classification | Not single-item classifier; adds complexity; licensing constraints; slower |
| **CLIP / Vision Feature Prints** | Vision Encoder (ViT-Tiny or native Vision API) | Pre-computed retrieval catalog | Apache 2.0 (CLIP) / Native (Vision API) | Similarity matching fallback (no retraining) | Zero-shot capable; no retraining needed; low latency with cached embeddings | Requires curated reference catalog; slower than direct classification; adds complexity |
| **watersplash/waste-classification** | Unknown (likely CNN) | General waste | Unknown | Community-referenced fallback | Referenced in multiple projects; general applicability | Unclear architecture; licensing unknown; untested |
| **MobileNetV2 (akmalia31 variant)** | MobileNetV2 CNN | 6 (TrashNet dataset) | Unknown | Legacy device support | Optimized for mobile; widely available; TrashNet benchmark | Older architecture; may lose accuracy vs newer models; licensing unclear |

---

## Recommendation Hierarchy

### **Phase 1: MVP (Recommended)**
**Start with: Recycling-Net-11**
- Pre-trained on waste data
- 11 categories sufficient for MVP
- Efficient, ANE-friendly
- Apache 2.0 licensed
- Clear conversion path to Core ML

### **Phase 2: Optimization (if needed)**
- **Higher accuracy required?** → Fine-tune **EfficientNet-Lite1** on your proprietary dataset
- **Multi-item scenes?** → Add **YOLOv8n** as optional detector layer
- **Fallback needed?** → Implement **Vision Feature Prints** for similarity-based lookup

### **Phase 3+: Advanced**
- **Scene complexity?** → Consider **MobileViT-XS** if clutter handling critical
- **Zero-shot capability?** → Integrate **CLIP** for open-vocabulary waste categories

---

## Decision Matrix

| Factor | Recycling-Net-11 | EfficientNet-Lite | MobileNetV3 | MobileViT-XS |
|---|---|---|---|---|
| Time to MVP | ✅ Immediate (pre-trained) | ❌ Needs fine-tuning | ❌ Needs fine-tuning | ❌ Needs fine-tuning |
| Inference Latency | ✅ <600ms | ✅ <600ms | ✅ <400ms | ⚠️ 800ms+ |
| ANE Support | ✅ Good | ✅ Excellent | ✅ Excellent | ⚠️ Limited (attention layers) |
| Accuracy (waste domain) | ✅ ~90% | ⚠️ Unknown (needs training) | ⚠️ Unknown (needs training) | ✅ ~92%+ (if trained well) |
| Model Size | ⚠️ ~100-200MB | ✅ ~50-100MB | ✅ <50MB | ⚠️ 150-250MB |
| Setup Effort | ✅ Minimal | ❌ Moderate | ❌ Moderate | ❌ High |
| License Risk | ✅ Apache 2.0 | ✅ Open | ✅ Open | ✅ Open |

---

## Recommended Action Plan

1. **Immediate**: Download & convert **Recycling-Net-11** to Core ML
   - Verify <600ms latency on iPhone 13+
   - Validate ≥90% accuracy on your test set

2. **If accuracy gap**: Fine-tune **EfficientNet-Lite1** on proprietary waste dataset
   - Use transfer learning for speed
   - Plan 2-3 week training cycle

3. **If multi-item scenes emerge**: Integrate **YOLOv8n** as optional pre-processor
   - Can be toggled on/off based on scene complexity
   - Add telemetry to measure adoption

4. **Fallback path**: Keep **MobileNetV3-Small** ready as emergency fallback
   - Can be trained quickly if primary model fails
   - Good compatibility with older devices

---

## Next Steps

Choose one:

**Option A (Fastest)**: Use Recycling-Net-11 → Run conversion script → Validate on device → Ship MVP

**Option B (Most Flexible)**: Set up fine-tuning pipeline for EfficientNet-Lite1 → Source waste dataset → Train → Convert → Validate

**Option C (Hybrid)**: Start with Recycling-Net-11 (MVP), prepare EfficientNet-Lite1 pipeline in parallel for Phase 2

Which path aligns with your timeline and accuracy requirements?
