"""Select probability and component thresholds on held-out observations."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from .data import find_data_root, group_views, read_grayscale, validation_filenames
from .inference import (
    hysteresis_to_labels,
    load_model,
    predict_probability,
    probability_to_labels,
)
from .metrics import PQCounts, score_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/kagglehub/competitions/filament-segmentation-2026")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/filament_unet.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration.json"))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--grow-thresholds", type=float, nargs="+", default=None,
                        help="If given, evaluate seeded growth at each lower threshold")
    parser.add_argument("--min-areas", type=int, nargs="+", default=[150, 300])
    parser.add_argument("--max-area", type=int, default=40000)
    parser.add_argument("--limb-margin", type=int, default=65)
    args = parser.parse_args()
    data = find_data_root(args.data)
    grouped = group_views(data.load_annotations())
    names = sorted(validation_filenames(grouped))[: args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tile_size = load_model(args.checkpoint, device)
    if model.split_manifest:
        raise ValueError('Use scripts/calibrate_checkpoint.py with --calibration-manifest for a custom-split checkpoint')
    grow_thresholds = args.grow_thresholds if args.grow_thresholds is not None else [None]
    choices = [
        (high, low, min_area)
        for high, low, min_area in itertools.product(args.thresholds, grow_thresholds, args.min_areas)
        if low is None or low <= high
    ]
    scores = {choice: PQCounts() for choice in choices}
    for index, name in enumerate(names, 1):
        image = read_grayscale(data.train_dir / name)
        probability = predict_probability(image, model, device, tile_size)
        for high, low, min_area in choices:
            if low is None:
                labels = probability_to_labels(image, probability, high, min_area,
                                               args.max_area, args.limb_margin)
            else:
                labels = hysteresis_to_labels(image, probability, high, low,
                                              min_area, args.max_area, args.limb_margin)
            scores[(high, low, min_area)] += score_observation(labels, grouped[name])
        if index % 5 == 0 or index == len(names):
            print(f"{index}/{len(names)} validation observations", flush=True)
    results = [
        {"threshold": high, "grow_threshold": low, "min_area": min_area,
         "pq": scores[(high, low, min_area)].pq,
         "tp": scores[(high, low, min_area)].tp,
         "fp": scores[(high, low, min_area)].fp,
         "fn": scores[(high, low, min_area)].fn}
        for high, low, min_area in choices
    ]
    results.sort(key=lambda row: row["pq"], reverse=True)
    payload = {"images": len(names), "checkpoint": str(args.checkpoint), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
