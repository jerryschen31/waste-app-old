#!/usr/bin/env python3
"""Download and assemble Food-101 using Hugging Face `datasets`.

Exports images into `data/{train,val,test}/food`. Creates a validation split
from the official train split by sampling `--val_fraction`.
"""

import argparse
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def ensure_dirs(base: Path):
    for split in ("train", "val", "test"):
        (base / split / "food").mkdir(parents=True, exist_ok=True)


def save_pil_image(img, out_path: Path):
    img = img.convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=92)


def export_list(records, out_dir: Path):
    i = 0
    for rec in records:
        label = rec.get("label_text") or rec.get("label")
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        img = rec["image"]
        # `img` may be a PIL Image or a dict with path
        if hasattr(img, "convert"):
            pil = img
        else:
            pil = Image.open(img["path"])
        fname = f"{label}__{i}.jpg"
        save_pil_image(pil, out_dir / fname)
        i += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data", help="Output data directory")
    p.add_argument("--val_fraction", type=float, default=0.1)
    args = p.parse_args()

    base = Path(args.data_dir)
    ensure_dirs(base)

    print("Streaming Food-101 via Hugging Face datasets (will download if needed)")

    # Use streaming to avoid materializing the full dataset in memory
    train_stream = load_dataset("food101", split="train", streaming=True)
    test_stream = load_dataset("food101", split="validation", streaming=True)

    train_out = base / "train" / "food"
    val_out = base / "val" / "food"
    test_out = base / "test" / "food"

    random.seed(42)
    # iterate train stream and randomly assign to val/train per val_fraction
    train_count = 0
    val_count = 0
    for i, rec in enumerate(train_stream):
        if random.random() < args.val_fraction:
            out_dir = val_out
            val_count += 1
        else:
            out_dir = train_out
            train_count += 1
        # rec['image'] could be a dict with 'bytes' or 'path' or a PIL Image
        img = rec["image"]
        label = rec.get("label_text") or rec.get("label")
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        if hasattr(img, "convert"):
            pil = img
        else:
            pil = Image.open(img["path"])
        fname = f"{label}__{i}.jpg"
        save_pil_image(pil, out_dir / fname)

    # iterate test stream
    test_count = 0
    for i, rec in enumerate(test_stream):
        img = rec["image"]
        label = rec.get("label_text") or rec.get("label")
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        if hasattr(img, "convert"):
            pil = img
        else:
            pil = Image.open(img["path"])
        fname = f"{label}__{i}.jpg"
        save_pil_image(pil, test_out / fname)
        test_count += 1

    print(f"Exported ~{train_count} train, {val_count} val, {test_count} test images")

    # small report
    report = {}
    for split in ("train", "val", "test"):
        folder = base / split / "food"
        files = sorted([p.name for p in folder.iterdir() if p.is_file()])
        report[split] = {"count": len(files), "samples": files[:10]}

    (base / "dataset_report.json").write_text(str(report))
    print("Wrote dataset_report.json")


if __name__ == "__main__":
    main()
