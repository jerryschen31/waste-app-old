#!/usr/bin/env python3
"""
Prepare the fine-Tuning dataset for RecyclingNet.

This script aggregates data from multiple sources, balances the classes,
and splits it into train/val/test sets for the 11-class fine-tuning task.

Sources:
- data/trash_items (Collected images)
- data/trash_augmented (Augmented images)
- data/kaggle_datasets (If applicable)

Output:
- data_finetune/
    train/
    val/
    test/
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict

# Configuration
SOURCE_DIRS = [
    Path("data/trash_items"),
    Path("data/trash_augmented"),
    # Add other sources as needed
]
OUTPUT_DIR = Path("data_finetune")
SPLIT_RATIOS = (0.8, 0.1, 0.1) # Train, Val, Test
SEED = 42

def scan_dataset(source_dirs):
    """Scans source directories for images organized by class."""
    image_paths = defaultdict(list)
    
    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f"Skipping missing source: {source_dir}")
            continue
            
        print(f"Scanning {source_dir}...")
        for class_dir in source_dir.iterdir():
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                class_name = class_dir.name
                images = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png"))
                
                image_paths[class_name].extend(images)
                
    return image_paths

def balance_dataset(image_paths, max_per_class=1000):
    """Optional: Limit the number of images per class to avoid imbalance."""
    balanced_paths = {}
    for class_name, paths in image_paths.items():
        if len(paths) > max_per_class:
            print(f"Downsampling {class_name}: {len(paths)} -> {max_per_class}")
            balanced_paths[class_name] = random.sample(paths, max_per_class)
        else:
            balanced_paths[class_name] = paths
    return balanced_paths

def create_splits(image_paths, output_dir, splits):
    """Copies images into train/val/test directories."""
    train_ratio, val_ratio, _ = splits
    
    # Clean output dir
    if output_dir.exists():
        print(f"Cleaning {output_dir}...")
        shutil.rmtree(output_dir)
    
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
        
    stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    
    for class_name, paths in tqdm(image_paths.items(), desc="Copying files"):
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_files = paths[:n_train]
        val_files = paths[n_train:n_train+n_val]
        test_files = paths[n_train+n_val:]
        
        for split_name, files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            split_dir = output_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for src_path in files:
                dst_path = split_dir / src_path.name
                # Avoid duplicate names if merging sources
                if dst_path.exists():
                    dst_path = split_dir / f"{src_path.stem}_{random.randint(0,9999)}{src_path.suffix}"
                    
                shutil.copy2(src_path, dst_path)
                stats[class_name][split_name] += 1
                
    return stats

def main():
    random.seed(SEED)
    
    print("Scanning datasets...")
    image_paths = scan_dataset(SOURCE_DIRS)
    
    total_images = sum(len(l) for l in image_paths.values())
    print(f"Found {total_images} total images across {len(image_paths)} classes.")
    
    # Optional: Balance dataset logic here
    # image_paths = balance_dataset(image_paths)
    
    print("Creating splits and copying files...")
    stats = create_splits(image_paths, OUTPUT_DIR, SPLIT_RATIOS)
    
    # Save statistics
    with open(OUTPUT_DIR / "dataset_report.json", "w") as f:
        json.dump(stats, f, indent=2)
        
    print(f"Dataset preparation complete. Saved to {OUTPUT_DIR}")
    print("Report saved to dataset_report.json")

if __name__ == "__main__":
    main()
