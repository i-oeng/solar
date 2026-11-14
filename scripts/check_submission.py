"""Verify every submission row and record the exact artifact hash."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

from solar_filament.data import find_data_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--data", default="data/kagglehub/competitions/filament-segmentation-2026")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    data = find_data_root(args.data)
    expected = {path.stem for path in data.test_dir.glob("*.jpeg")}
    seen = set()
    represented = set()
    occupied = {}
    with args.submission.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"Unexpected columns: {reader.fieldnames}")
        for row in reader:
            identifier = row["filament_id"]
            stem, instance = identifier.rsplit("_", 1)
            if identifier in seen or stem not in expected or int(instance) <= 0:
                raise ValueError(f"Invalid or duplicate instance ID: {identifier}")
            rle = {"size": [2048, 2048], "counts": row["segmentation_rle"].encode("ascii")}
            mask = coco_mask.decode(rle)
            if mask.shape != (2048, 2048) or not mask.any():
                raise ValueError(f"Empty or malformed mask: {identifier}")
            packed = np.packbits(mask.reshape(-1))
            if stem in occupied:
                if np.any(occupied[stem] & packed):
                    raise ValueError(f"Overlapping instances: {identifier}")
                occupied[stem] |= packed
            else:
                occupied[stem] = packed
            seen.add(identifier)
            represented.add(stem)
    if not seen:
        raise ValueError("Submission has no instances")
    payload = {
        "submission": str(args.submission), "rows": len(seen),
        "test_images": len(expected), "images_with_predictions": len(represented),
        "images_without_predictions": sorted(expected - represented),
        "all_masks_decoded": True,
        "instances_nonoverlapping": True,
        "bytes": args.submission.stat().st_size,
        "sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest(),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
