#!/usr/bin/env python3
"""Validate and clean dataset images, then write `data/dataset_report.json`.

Deletes files that PIL cannot open/read, and produces a JSON report with counts
and sample filenames per split/label.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


def is_image_readable(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        # Re-open to ensure full decode
        with Image.open(p) as im:
            im.copy()
        return True
    except Exception:
        return False


def clean_and_report(base: Path, report_path: Path, max_samples=10):
    removed = 0
    report = {}
    for split in ("train", "val", "test"):
        report[split] = {}
        for label in ("food", "not_food"):
            folder = base / split / label
            if not folder.exists():
                report[split][label] = {"count": 0, "samples": []}
                continue
            files = sorted([p for p in folder.iterdir() if p.is_file()])
            good = []
            for f in files:
                if is_image_readable(f):
                    good.append(f.name)
                else:
                    try:
                        f.unlink()
                        removed += 1
                    except Exception:
                        pass
            report[split][label] = {"count": len(good), "samples": good[:max_samples]}

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return removed, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data", help="Root data directory")
    p.add_argument("--report", default="data/dataset_report.json", help="Path to write report JSON")
    args = p.parse_args()

    base = Path(args.data_dir)
    report_path = Path(args.report)
    print(f"Scanning and cleaning dataset under {base}")
    removed, report = clean_and_report(base, report_path)
    print(f"Removed {removed} unreadable/corrupt images")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
