#!/usr/bin/env python3
"""Create a balanced subset dataset: 10k positives and 10k negatives with 80/10/10 split.

This script samples from `data/*/food` for positives and `data/*/not_food` for negatives
and writes the balanced dataset into `data_balanced/{train,val,test}/{food,not_food}`.

Usage:
  source ml_env/bin/activate
  python3 scripts/make_balanced_subset.py --data_dir data --out_dir data_balanced --pos_total 10000 --neg_total 10000
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image


def find_files(root: Path, pattern="*"):
    files = []
    for p in root.rglob(pattern):
        if p.is_file():
            files.append(p)
    return files


def safe_copy_convert(src: Path, dst: Path):
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, format="JPEG", quality=92)
        return True
    except Exception as e:
        # fallback to shutil copy
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception:
            return False


def sample_and_write(files, counts, out_dirs, prefix):
    # files: list[Path], counts: (n_train,n_val,n_test), out_dirs: dict
    random.shuffle(files)
    n_train, n_val, n_test = counts
    i = 0
    written = 0
    for dst_dir, n in zip((out_dirs['train'], out_dirs['val'], out_dirs['test']), (n_train, n_val, n_test)):
        for j in range(n):
            if i >= len(files):
                raise ValueError("Not enough files to sample")
            src = files[i]
            dst = dst_dir / f"{prefix}__{src.stem}.jpg"
            ok = safe_copy_convert(src, dst)
            if ok:
                written += 1
            i += 1
    return written


def make_report(out_base: Path, report_path: Path, max_samples=10):
    report = {}
    for split in ("train", "val", "test"):
        report[split] = {}
        for label in ("food", "not_food"):
            folder = out_base / split / label
            files = sorted([p.name for p in folder.iterdir() if p.is_file()]) if folder.exists() else []
            report[split][label] = {"count": len(files), "samples": files[:max_samples]}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data", help="Source data dir (has train/val/test food and not_food)")
    p.add_argument("--out_dir", default="data_balanced", help="Output balanced dataset dir")
    p.add_argument("--pos_total", type=int, default=10000)
    p.add_argument("--neg_total", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    src = Path(args.data_dir)
    out = Path(args.out_dir)

    # gather positives and negatives from all splits
    pos_files = []
    neg_files = []
    for split in ("train", "val", "test"):
        pos_folder = src / split / "food"
        neg_folder = src / split / "not_food"
        if pos_folder.exists():
            pos_files.extend([p for p in pos_folder.iterdir() if p.is_file()])
        if neg_folder.exists():
            neg_files.extend([p for p in neg_folder.iterdir() if p.is_file()])

    if len(pos_files) < args.pos_total:
        raise RuntimeError(f"Not enough positive images: have {len(pos_files)}, need {args.pos_total}")
    if len(neg_files) < args.neg_total:
        raise RuntimeError(f"Not enough negative images: have {len(neg_files)}, need {args.neg_total}")

    # compute splits (80/10/10)
    def splits_for(n):
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        n_test = n - n_train - n_val
        return (n_train, n_val, n_test)

    pos_counts = splits_for(args.pos_total)
    neg_counts = splits_for(args.neg_total)

    out_dirs_pos = {s: out / s / "food" for s in ("train", "val", "test")}
    out_dirs_neg = {s: out / s / "not_food" for s in ("train", "val", "test")}

    # sample and write
    written_pos = sample_and_write(pos_files, pos_counts, out_dirs_pos, "pos")
    written_neg = sample_and_write(neg_files, neg_counts, out_dirs_neg, "neg")

    report = make_report(out, out / "dataset_report.json")
    print(f"Wrote balanced dataset to {out} (pos {written_pos}, neg {written_neg})")


if __name__ == "__main__":
    main()
