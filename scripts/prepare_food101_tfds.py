#!/usr/bin/env python3
"""Download and assemble Food-101 using TensorFlow Datasets.

This script uses `tensorflow_datasets` to fetch the Food-101 dataset and
exports images into `data/{train,val,test}/food`. It creates a validation
split by sampling a fraction of the train set.

Run:
  source ml_env/bin/activate
  python3 scripts/prepare_food101_tfds.py --data_dir data --val_fraction 0.1
"""

import argparse
import os
import random
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image


def ensure_dirs(base):
    for split in ("train", "val", "test"):
        (base / split / "food").mkdir(parents=True, exist_ok=True)


def save_image(tf_image, path: Path):
    # tf_image is a uint8 Tensor
    arr = tf_image.numpy()
    im = Image.fromarray(arr)
    im = im.convert("RGB")
    im.save(path, format="JPEG", quality=92)


def export_split(ds, out_dir: Path, prefix=""):
    i = 0
    for ex in tfds.as_numpy(ds):
        label_text = ex["label_text"].decode("utf-8") if isinstance(ex["label_text"], bytes) else ex["label_text"]
        img = tf.convert_to_tensor(ex["image"])
        fname = f"{label_text}__{i}.jpg"
        save_image(img, out_dir / fname)
        i += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data", help="Output data directory")
    p.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of train to use for validation")
    args = p.parse_args()

    base = Path(args.data_dir)
    ensure_dirs(base)

    print("Loading Food-101 via TFDS (this will download the dataset if needed)")
    # tfds splits: 'train' and 'test'
    train_ds = tfds.load("food101", split="train", shuffle_files=True, as_supervised=False)
    test_ds = tfds.load("food101", split="test", as_supervised=False)

    # Materialize train into list for splitting (memory OK ~75k images)
    train_list = list(tfds.as_numpy(train_ds))
    random.seed(42)
    random.shuffle(train_list)
    n_val = int(len(train_list) * args.val_fraction)
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    print(f"Exporting {len(train_list)} train images, {len(val_list)} val images, {sum(1 for _ in tfds.as_numpy(test_ds))} test images")

    # Export
    export_split(train_list, base / "train" / "food")
    export_split(val_list, base / "val" / "food")
    # For test, tfds.as_numpy returns a generator; re-load to iterate
    test_ds = tfds.load("food101", split="test", as_supervised=False)
    export_split(test_ds, base / "test" / "food")

    # Write a small report
    report = {}
    for split in ("train", "val", "test"):
        folder = base / split / "food"
        files = sorted([p.name for p in folder.iterdir() if p.is_file()])
        report[split] = {"count": len(files), "samples": files[:10]}

    (base / "dataset_report.json").write_text(str(report))
    print("Wrote dataset_report.json")


if __name__ == "__main__":
    main()
