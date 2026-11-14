"""Cache lossless binary training masks for fast patch training."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .data import annotation_mask, find_data_root, group_views


def choose_view(views: list[list[dict]]) -> list[dict]:
    """Select one complete annotation view per physical observation."""
    return max(views, key=lambda view: (len(view), sum(float(a["area"]) for a in view)))


def prepare_masks(data_path: str | Path, output_dir: Path) -> None:
    data = find_data_root(data_path)
    grouped = group_views(data.load_annotations())
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (filename, views) in enumerate(sorted(grouped.items()), 1):
        output = output_dir / f"{Path(filename).stem}.png"
        if output.exists():
            continue
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        for annotation in choose_view(views):
            mask[annotation_mask(annotation)] = 255
        if not cv2.imwrite(str(output), mask):
            raise OSError(f"Could not write {output}")
        if index % 50 == 0 or index == len(grouped):
            print(f"{index}/{len(grouped)} masks prepared", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/kagglehub/competitions/filament-segmentation-2026")
    parser.add_argument("--output", type=Path, default=Path("data/prepared/masks"))
    args = parser.parse_args()
    prepare_masks(args.data, args.output)


if __name__ == "__main__":
    main()

