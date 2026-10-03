# Waste Photo Dataset Curation Strategy

## Objective
Curate ~2000 high-quality, single-object waste images for testing the waste classification models (FoodDetector cascade + RecyclingNet11). Images should represent realistic items users might dispose in trash, recycle, compost, or e-waste bins.

## Target Distribution
- **Trash**: ~400-500 images (plastic bags, styrofoam, broken items, general waste)
- **Recycle**: ~500-600 images (paper, cardboard, plastic bottles, aluminum cans, glass)
- **Compost**: ~400-500 images (food scraps, leaves, plant matter, coffee grounds)
- **E-waste**: ~300-400 images (old phones, cables, circuit boards, batteries, headphones)
- **Other/Ambiguous**: ~200 images (items that could go multiple places, edge cases)

**Total**: ~1800-2000 images

## Free Online Sources

### Primary Sources (Recommended)
1. **Unsplash** (unsplash.com) - High quality, no attribution required
   - Search terms: "trash bin", "recycling bin", "plastic bottle", "cardboard box", "electronic waste", etc.
   - Download API available: https://api.unsplash.com/search/photos
   
2. **Pexels** (pexels.com) - Large free library, creative commons
   - Search terms: similar to Unsplash
   - Download via Python: requests + pagination

3. **Pixabay** (pixabay.com) - Diverse content, free commercial use
   - Search terms: waste, trash, recycling, compost, electronic waste
   - API available: https://pixabay.com/api/

4. **Wikipedia Commons** (commons.wikimedia.org) - Educational images, various licenses
   - Search: waste management, recycling symbols, e-waste categories

### Secondary Sources
5. **Product/Store Photos** - eBay, Amazon, Mercari for specific item types
   - Single objects already cropped/isolated
   - Good for e-waste (headphones, cables, laptops, etc.)

6. **Creative Commons Search** (search.creativecommons.org)
   - Filter by license type and source

## Download Strategy

### Option A: Manual Curation (Recommended for V1)
1. Visit Unsplash.com, Pexels.com, Pixabay.com
2. Search for specific waste items (e.g., "plastic bottle", "aluminum can", "cardboard", "old phone")
3. Download top ~20-30 results per search
4. Organize into category folders
5. Quick quality check (single object, clear background, not blurry)

**Estimated time**: 2-3 hours for ~2000 images

### Option B: Automated Download with Python
1. Use `scripts/download_waste_images.py` (to be created)
2. Script queries multiple free APIs (Unsplash, Pexels, Pixabay)
3. Automatically filters for single objects
4. Organizes into category directories
5. Logs sources and licenses

**Estimated time**: 30-60 minutes (setup) + runtime

## Quality Criteria for Images

✅ **Accept**:
- Single object (clear, isolated)
- Good lighting (no extreme shadows/glare)
- Clear background (blurred or plain)
- Recognizable waste item
- Reasonable resolution (>200x200 pixels)
- No watermarks (unless very subtle)

❌ **Reject**:
- Blurry or low resolution
- Multiple objects clustered
- Extreme angles (hard to identify)
- Person/hand visible holding item
- Extremely dark or blown-out
- Watermarked stock photos

## Directory Structure

```
data/waste_test/
├── trash/
│   ├── item_001.jpg
│   ├── item_002.jpg
│   └── ... (300-500 images)
├── recycle/
│   ├── bottle_001.jpg
│   ├── can_001.jpg
│   └── ... (500-600 images)
├── compost/
│   ├── food_waste_001.jpg
│   ├── leaves_001.jpg
│   └── ... (300-400 images)
├── e_waste/
│   ├── phone_001.jpg
│   ├── cable_001.jpg
│   └── ... (300-400 images)
├── other/
│   ├── ambiguous_001.jpg
│   └── ... (200 images)
└── SOURCES.md  ← Document all sources, licenses, and download dates
```

## Specific Items to Target (per category)

### Trash (~400-500)
- Plastic bags, wrappers, packaging
- Styrofoam, foam padding
- Broken plastic toys
- Used aluminum foil
- Used tissues, paper towels
- Broken ceramics, pottery shards
- Worn-out sponges

### Recycle (~500-600)
- Plastic bottles (clear, colored, various sizes)
- Aluminum cans, steel cans
- Glass bottles, glass jars
- Cardboard boxes, paper boxes
- Newspapers, magazines, junk mail
- Metal cans (food cans, paint cans)
- Plastic containers, takeout boxes
- Paper bags

### Compost (~300-400)
- Apple cores, banana peels
- Vegetable scraps (carrot peels, onion skins)
- Coffee grounds, tea bags
- Bread, pasta
- Fallen leaves, twigs
- Wilted flowers
- Grass clippings

### E-waste (~300-400)
- Old smartphones
- USB cables, charging cables
- Computer circuit boards
- Old headphones, earbuds
- Laptop batteries
- Broken hard drives
- Old monitors (front/back view)
- Electronic control boards

## Implementation Steps

1. **Create directory structure** → `mkdir -p data/waste_test/{trash,recycle,compost,e_waste,other}`

2. **Option A (Manual)**: Spend 2-3 hours downloading via Unsplash/Pexels/Pixabay

3. **Option B (Semi-automated)**:
   - Run `scripts/download_waste_images.py` to download bulk via APIs
   - Manually review and move images to correct category folders
   - Remove duplicates and low-quality items

4. **Quality Check**:
   - Spot-check each category (review 20-30 images per category)
   - Remove rejected images

5. **Document**:
   - Create `data/waste_test/SOURCES.md` listing all sources, download dates, license types
   - Log any custom curation notes

6. **Validate**:
   - Count images per category
   - Run `verify-waste-dataset.py` to check image dimensions, formats
   - Spot-check for duplicates

## Next: Testing

Once dataset is ready:
1. Run inference on all images with cascade (FoodDetector → RecyclingNet11)
2. Measure per-class accuracy
3. Identify weak categories (low accuracy) → gather more of those images
4. Validate model generalization (diversity of angles, lighting, backgrounds)

## Timeline
- **Phase 1 (V1)**: Manual curation of ~2000 images (2-3 hours)
- **Phase 2 (V2)**: Automated download script, de-duplication, filtering
- **Phase 3 (V3)**: Augmentation, edge case collection, per-class accuracy optimization

---

**Status**: Ready for manual curation (Option A) or Python script development (Option B)

**Owner**: Current agent

**Last Updated**: 2026-01-07
