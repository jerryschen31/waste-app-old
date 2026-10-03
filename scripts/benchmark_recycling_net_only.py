#!/usr/bin/env python3
"""
Benchmark RecyclingNet11ex ONLY (No Cascade).
"""
import coremltools as ct
from PIL import Image
from pathlib import Path
from tqdm import tqdm

TEST_DATA_DIR = Path("data/waste_test_filtered")
MODEL_PATH = "models/coreml/RecyclingNet11ex.mlpackage"

RN_TO_FINAL_CATEGORY = {
    'biological': 'Compost',
    'cardboard': 'Recycle',
    'clothes': 'Trash',
    'electronics': 'e-Waste',
    'glass': 'Recycle',
    'metal': 'Recycle',
    'other': 'Trash',
    'paper': 'Recycle',
    'plastic': 'Recycle',
    'shoes': 'Trash',
    'trash': 'Trash'
}

def main():
    print(f"Loading {MODEL_PATH}...")
    model = ct.models.MLModel(MODEL_PATH)
    
    total = 0
    correct = 0
    
    files = []
    for category in TEST_DATA_DIR.iterdir():
        if category.is_dir():
            for img_path in category.glob("*.jpg"):
                files.append((category.name, img_path))
                
    print(f"Testing {len(files)} images...")
    
    for true_category, img_path in tqdm(files):
        # Normalize true category
        if true_category == "e_waste": true_category = "e-Waste"
        if true_category == "compost": true_category = "Compost"
        if true_category == "recycle": true_category = "Recycle"
        if true_category == "trash": true_category = "Trash"
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize((224, 224))
            
            # Predict
            pred = model.predict({'input': image})
            
            # Get class
            probs = pred['classLabel_probs']
            rn_class = max(probs, key=probs.get)
            
            final_pred = RN_TO_FINAL_CATEGORY.get(rn_class, "Trash")
            
            if final_pred == true_category:
                correct += 1
            total += 1
            
        except Exception as e:
            print(f"Error {img_path}: {e}")
            
    print(f"Accuracy: {correct/total:.2%}")

if __name__ == "__main__":
    main()
