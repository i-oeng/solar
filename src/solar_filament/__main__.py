"""Run data checks, local validation, and submission generation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

from .baseline import BaselineConfig, predict_labels
from .data import find_data_root, group_views, read_grayscale, validation_filenames
from .metrics import PQCounts, score_observation


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Solar Filament Segmentation Challenge 2026")
    cli.add_argument(
        "--data",
        default="data/kagglehub/competitions/filament-segmentation-2026",
        help="KaggleHub download directory or MAGFiLO_1.0_Kaggle_2026 directory",
    )
    subcommands = cli.add_subparsers(dest="command", required=True)
    subcommands.add_parser("inspect", help="Describe images and annotation views")
    for name in ("validate", "submit"):
        subcommand = subcommands.add_parser(name)
        subcommand.add_argument("--threshold", type=float, default=14.0)
        subcommand.add_argument("--sigma", type=float, default=24.0)
        subcommand.add_argument("--min-area", type=int, default=250)
        subcommand.add_argument("--max-area", type=int, default=40000)
        subcommand.add_argument("--limb-margin", type=int, default=65)
        subcommand.add_argument("--close-size", type=int, default=3)
        subcommand.add_argument("--limit", type=int, default=None)
        if name == "submit":
            subcommand.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    return cli


def config_from_args(args: argparse.Namespace) -> BaselineConfig:
    return BaselineConfig(
        contrast_threshold=args.threshold,
        background_sigma=args.sigma,
        min_area=args.min_area,
        max_area=args.max_area,
        limb_margin=args.limb_margin,
        close_size=args.close_size,
    )


def inspect(data) -> None:
    metadata = data.load_annotations()
    grouped = group_views(metadata)
    validation = validation_filenames(grouped)
    details = {
        "physical_train_images": len(grouped),
        "annotation_views": len(metadata["images"]),
        "annotations": len(metadata["annotations"]),
        "test_images": len(list(data.test_dir.glob("*.jpeg"))),
        "validation_physical_images": len(validation),
        "train_physical_images": len(grouped) - len(validation),
    }
    print(json.dumps(details, indent=2))


def validate(data, config: BaselineConfig, limit: int | None) -> None:
    grouped = group_views(data.load_annotations())
    names = sorted(validation_filenames(grouped))
    if limit is not None:
        names = names[:limit]
    total = PQCounts()
    start = time.monotonic()
    for index, name in enumerate(names, 1):
        image = read_grayscale(data.train_dir / name)
        labels = predict_labels(image, config)
        total += score_observation(labels, grouped[name])
        if index % 10 == 0 or index == len(names):
            print(f"{index}/{len(names)} images | PQ={total.pq:.4f} TP={total.tp} FP={total.fp} FN={total.fn}", flush=True)
    print(json.dumps({
        "images": len(names),
        "annotator_views": sum(len(grouped[name]) for name in names),
        "pq": total.pq,
        "tp": total.tp,
        "fp": total.fp,
        "fn": total.fn,
        "seconds": round(time.monotonic() - start, 2),
        "config": config.__dict__,
    }, indent=2))


def submit(data, config: BaselineConfig, limit: int | None, output: Path) -> None:
    images = sorted(data.test_dir.glob("*.jpeg"))
    if limit is not None:
        images = images[:limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    start = time.monotonic()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for index, path in enumerate(images, 1):
            labels = predict_labels(read_grayscale(path), config)
            for label in range(1, int(labels.max()) + 1):
                mask = np.asfortranarray((labels == label).astype(np.uint8))
                encoded = coco_mask.encode(mask)["counts"].decode("ascii")
                writer.writerow([f"{path.stem}_{label}", encoded])
                rows += 1
            if index % 10 == 0 or index == len(images):
                print(f"{index}/{len(images)} images | {rows} masks", flush=True)
    print(json.dumps({"output": str(output), "images": len(images), "masks": rows,
                      "seconds": round(time.monotonic() - start, 2)}, indent=2))


def main() -> None:
    args = parser().parse_args()
    data = find_data_root(args.data)
    if args.command == "inspect":
        inspect(data)
    elif args.command == "validate":
        validate(data, config_from_args(args), args.limit)
    elif args.command == "submit":
        submit(data, config_from_args(args), args.limit, args.output)


if __name__ == "__main__":
    main()

