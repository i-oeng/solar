"""Tile-based neural inference and instance conversion at native resolution."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask

from .baseline import components_to_labels, solar_disk_mask
from .data import find_data_root, group_views, read_grayscale, validation_filenames
from .metrics import PQCounts, score_observation
from .model import FilamentUNet, build_model
from .train import make_features
from .features import FEATURE_SPECS


def tile_starts(length: int, size: int, stride: int) -> list[int]:
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[FilamentUNet, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint.get('architecture', 'compact'), base=checkpoint["base_channels"])
    model.load_state_dict(checkpoint["model"])
    recipe = checkpoint.get('feature_recipe', 'contrast')
    if recipe not in FEATURE_SPECS or checkpoint.get('feature_spec', FEATURE_SPECS[recipe]) != FEATURE_SPECS[recipe]:
        raise ValueError('Unknown or incompatible checkpoint feature recipe')
    model.feature_recipe = recipe
    model.split_manifest = checkpoint.get('split_manifest')
    model.to(device).eval()
    return model, checkpoint["crop_size"]


@torch.inference_mode()
def predict_probability(
    image: np.ndarray, model: FilamentUNet, device: torch.device,
    tile_size: int, overlap: int = 128, batch_size: int = 2,
) -> np.ndarray:
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile size")
    coordinates = [(y, x) for y in tile_starts(image.shape[0], tile_size, stride)
                   for x in tile_starts(image.shape[1], tile_size, stride)]
    total = np.zeros(image.shape, dtype=np.float32)
    weight_sum = np.zeros(image.shape, dtype=np.float32)
    axis_weight = np.maximum(np.hanning(tile_size).astype(np.float32), 0.1)
    weight = axis_weight[:, None] * axis_weight[None, :]
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start : start + batch_size]
        features = np.stack([
            make_features(image[y : y + tile_size, x : x + tile_size], getattr(model, 'feature_recipe', 'contrast'))
            for y, x in batch_coordinates
        ])
        tensor = torch.from_numpy(features).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probabilities = torch.sigmoid(model(tensor)).float().cpu().numpy()[:, 0]
        for (y, x), probability in zip(batch_coordinates, probabilities):
            total[y : y + tile_size, x : x + tile_size] += probability * weight
            weight_sum[y : y + tile_size, x : x + tile_size] += weight
    return total / weight_sum


def probability_to_labels(
    image: np.ndarray, probability: np.ndarray,
    threshold: float, min_area: int, max_area: int, limb_margin: int,
) -> np.ndarray:
    disk = solar_disk_mask(image, limb_margin)
    binary = ((probability >= threshold) & disk).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return components_to_labels(binary, min_area, max_area)


def hysteresis_to_labels(
    image: np.ndarray, probability: np.ndarray, high_threshold: float,
    low_threshold: float, min_seed_area: int, max_area: int, limb_margin: int,
) -> np.ndarray:
    """Grow high-confidence instances into connected lower-confidence pixels.

    Components without a sufficiently large high-confidence seed are discarded.
    This can recover faint filament tails without admitting every weak texture.
    """
    if not 0 <= low_threshold <= high_threshold <= 1:
        raise ValueError("Require 0 <= low_threshold <= high_threshold <= 1")
    disk = solar_disk_mask(image, limb_margin)
    strong = ((probability >= high_threshold) & disk).astype(np.uint8)
    seeds = components_to_labels(strong, min_seed_area, max_area)
    weak = ((probability >= low_threshold) & disk).astype(np.uint8)
    weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, components, stats, _ = cv2.connectedComponentsWithStats(weak, 8)
    seeded = np.bincount(components[seeds > 0], minlength=count) > 0
    valid = np.flatnonzero(seeded & (stats[:, cv2.CC_STAT_AREA] <= max_area))
    valid = valid[valid != 0]
    lookup = np.zeros(count, dtype=np.int32)
    lookup[valid] = np.arange(1, len(valid) + 1)
    return lookup[components]


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("command", choices=("validate", "submit"))
    cli.add_argument("--data", default="data/kagglehub/competitions/filament-segmentation-2026")
    cli.add_argument("--checkpoint", type=Path, default=Path("checkpoints/filament_unet.pt"))
    cli.add_argument("--output", type=Path, default=Path("outputs/unet_submission.csv"))
    cli.add_argument("--threshold", type=float, default=0.5)
    cli.add_argument("--grow-threshold", type=float, default=None,
                     help="If set, grow high-threshold seeds into lower-confidence components")
    cli.add_argument("--min-area", type=int, default=150)
    cli.add_argument("--max-area", type=int, default=40000)
    cli.add_argument("--limb-margin", type=int, default=65)
    cli.add_argument("--overlap", type=int, default=128)
    cli.add_argument("--batch-size", type=int, default=2)
    cli.add_argument("--limit", type=int, default=None)
    args = cli.parse_args()
    data = find_data_root(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tile_size = load_model(args.checkpoint, device)
    if args.command == 'validate' and model.split_manifest:
        raise ValueError('Use scripts/evaluate_checkpoint.py to evaluate a custom-split checkpoint with its recorded validation observations')
    print(f"Inference device: {device}", flush=True)
    started = time.monotonic()
    def labels_for(path: Path) -> np.ndarray:
        image = read_grayscale(path)
        probability = predict_probability(image, model, device, tile_size,
                                          args.overlap, args.batch_size)
        if args.grow_threshold is not None:
            return hysteresis_to_labels(image, probability, args.threshold,
                                        args.grow_threshold, args.min_area,
                                        args.max_area, args.limb_margin)
        return probability_to_labels(image, probability, args.threshold,
                                     args.min_area, args.max_area, args.limb_margin)
    if args.command == "validate":
        grouped = group_views(data.load_annotations())
        names = sorted(validation_filenames(grouped))
        if args.limit is not None:
            names = names[: args.limit]
        total = PQCounts()
        for index, name in enumerate(names, 1):
            total += score_observation(labels_for(data.train_dir / name), grouped[name])
            if index % 10 == 0 or index == len(names):
                print(f"{index}/{len(names)} images | PQ={total.pq:.4f} TP={total.tp} FP={total.fp} FN={total.fn}", flush=True)
        print(json.dumps({"images": len(names), "pq": total.pq, "tp": total.tp,
                          "fp": total.fp, "fn": total.fn,
                          "seconds": round(time.monotonic() - started, 1)}, indent=2))
    else:
        images = sorted(data.test_dir.glob("*.jpeg"))
        if args.limit is not None:
            images = images[: args.limit]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rows = 0
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filament_id", "segmentation_rle"])
            for index, path in enumerate(images, 1):
                labels = labels_for(path)
                for instance in range(1, int(labels.max()) + 1):
                    rle = coco_mask.encode(np.asfortranarray((labels == instance).astype(np.uint8)))
                    writer.writerow([f"{path.stem}_{instance}", rle["counts"].decode("ascii")])
                    rows += 1
                if index % 10 == 0 or index == len(images):
                    print(f"{index}/{len(images)} images | {rows} masks", flush=True)
        print(json.dumps({"output": str(args.output), "masks": rows,
                          "seconds": round(time.monotonic() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
