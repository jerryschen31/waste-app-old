#!/usr/bin/env python3
"""Prepare datasets for fine-tuning EfficientNet-B2.

Features:
- Downloads Food-101 (if not already downloaded) and extracts into `data/`
- Assembles positives into `data/{train,val,test}/food` using Food-101 splits
- Optionally samples negatives from Open Images (requires `fiftyone`)
- Converts images to JPEG RGB and validates files
- Writes `dataset_report.json` with counts and sample filenames

Notes:
- This script does not download the entire Open Images dataset by default. Use `--negatives N` to request N negatives; the script will try to use FiftyOne to fetch them. If FiftyOne is not installed, negatives step will be skipped and instructions will be printed.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

from zipfile import ZipFile

try:
    from PIL import Image
except Exception:
    print("Please install Pillow: pip install Pillow")
    raise


FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.zip"


def download_file(url, dest_path, show_progress=True):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"Found existing {dest_path}")
        return dest_path
    print(f"Downloading {url} -> {dest_path}")
    urlretrieve(url, dest_path)
    return dest_path


def extract_zip(zip_path, extract_to):
    with ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


def make_dirs(base):
    for split in ("train", "val", "test"):
        (base / split / "food").mkdir(parents=True, exist_ok=True)
        (base / split / "not_food").mkdir(parents=True, exist_ok=True)


def safe_convert_image(src_path, dst_path, target_format="JPEG"):
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst_path, format=target_format, quality=92)
        return True
    except Exception as e:
        print(f"Failed to convert {src_path}: {e}")
        return False


def assemble_food101(raw_dir: Path, out_base: Path, val_fraction=0.1):
    # raw_dir should contain 'food-101' extracted structure
    images_root = raw_dir / "food-101" / "images"
    meta_root = raw_dir / "food-101" / "meta"
    if not images_root.exists() or not meta_root.exists():
        raise FileNotFoundError("Expected Food-101 structure not found in raw dir")

    # read train/test splits
    train_txt = meta_root / "train.txt"
    test_txt = meta_root / "test.txt"

    train_items = [l.strip() for l in train_txt.read_text().splitlines() if l.strip()]
    test_items = [l.strip() for l in test_txt.read_text().splitlines() if l.strip()]

    # prepare mapping: each item like 'class/image'
    out_train = out_base / "train" / "food"
    out_val = out_base / "val" / "food"
    out_test = out_base / "test" / "food"

    # ensure dirs
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    import random

    random.seed(42)

    # Split train -> train/val
    for item in train_items:
        class_name, imgname = item.split("/")
        src = images_root / class_name / (imgname + ".jpg")
        # sample into val with probability val_fraction
        if random.random() < val_fraction:
            dst = out_val / f"{class_name}__{imgname}.jpg"
        else:
            dst = out_train / f"{class_name}__{imgname}.jpg"
        safe_convert_image(src, dst)

    for item in test_items:
        class_name, imgname = item.split("/")
        src = images_root / class_name / (imgname + ".jpg")
        dst = out_test / f"{class_name}__{imgname}.jpg"
        safe_convert_image(src, dst)


def sample_openimages_negatives(out_base: Path, n_samples=2000):
    """Attempt to sample negatives using FiftyOne. If not available, return False.

    This will download images into `out_base/{train,val,test}/not_food` sampling proportionally.
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except Exception as e:
        print("FiftyOne not available; skipping Open Images negative sampling.")
        print("To enable, install: pip install fiftyone")
        return False

    print("Downloading a sampled Open Images subset via FiftyOne (may take time)...")
    # Load a sample of open-images-v6 train split
    samples = int(n_samples)
    dataset = foz.load_zoo_dataset("open-images-v6", split="train", max_samples=samples, shuffle=True)

    # Export images into our not_food folders
    per_split = max(1, samples // 3)
    splits = [("train", per_split), ("val", per_split), ("test", samples - 2 * per_split)]
    out_dirs = {"train": out_base / "train" / "not_food", "val": out_base / "val" / "not_food", "test": out_base / "test" / "not_food"}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Iterate samples from the FiftyOne dataset in order and distribute into splits
    i = 0
    cur_split_idx = 0
    split_name, split_cnt = splits[cur_split_idx]
    written = 0
    for sample in dataset.take(samples):
        if i >= split_cnt:
            i = 0
            cur_split_idx += 1
            split_name, split_cnt = splits[cur_split_idx]
        src = sample.filepath
        dst = out_dirs[split_name] / Path(src).name
        safe_convert_image(src, dst)
        i += 1
        written += 1

    print(f"Exported {written} negative images into data/*/not_food")

    # Clean up if FiftyOne created a dataset in user's default location
    try:
        dataset.delete()
    except Exception:
        pass
    return True


def make_report(out_base: Path, report_path: Path, max_samples=10):
    report = {}
    for split in ("train", "val", "test"):
        report[split] = {}
        for label in ("food", "not_food"):
            folder = out_base / split / label
            if not folder.exists():
                report[split][label] = {"count": 0, "samples": []}
                continue
            files = sorted([p.name for p in folder.iterdir() if p.is_file()])
            report[split][label] = {"count": len(files), "samples": files[:max_samples]}

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote dataset report to {report_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data", help="Output data directory (will be created)")
    p.add_argument("--raw_dir", default="data/raw", help="Where to store downloaded zips and extracted raw files")
    p.add_argument("--download_food101", action="store_true", help="Download Food-101 archive if missing")
    p.add_argument("--negatives", type=int, default=0, help="Number of negative images to sample from Open Images (optional, requires FiftyOne)")
    p.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of train to use as validation")
    args = p.parse_args()

    base = Path(args.data_dir)
    raw = Path(args.raw_dir)
    make_dirs(base)

    if args.download_food101:
        zip_path = raw / "food-101.zip"
        download_file(FOOD101_URL, zip_path)
        # extract
        extract_dir = raw
        print("Extracting Food-101")
        extract_zip(zip_path, extract_dir)
    else:
        if not (raw / "food-101").exists():
            # Allow proceeding if data already assembled (from HF path)
            if (base / "train" / "food").exists():
                print("Food-101 raw extraction not found, but `data/train/food` exists — skipping assembly")
            else:
                print("Food-101 extraction not found in raw dir. Re-run with --download_food101")
                sys.exit(1)
        else:
            print("Assembling Food-101 into data folders (converting to RGB JPEG)")
            assemble_food101(raw, base, val_fraction=args.val_fraction)

    if args.negatives and args.negatives > 0:
        ok = sample_openimages_negatives(base, n_samples=args.negatives)
        if not ok:
            print("Negative sampling was skipped. To fetch negatives, install FiftyOne and re-run with --negatives N")

    # generate report
    report_path = base / "dataset_report.json"
    make_report(base, report_path)


if __name__ == "__main__":
    main()
