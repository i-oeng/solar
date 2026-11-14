"""Compare checkpoints at frozen settings, including object-size recall."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_boundary_growth import grow_boundaries
from checkpoint_protocol import checkpoint_split
from solar_filament.data import annotation_mask, find_data_root, group_views, read_grayscale, validation_filenames
from solar_filament.inference import hysteresis_to_labels, load_model, predict_probability
from solar_filament.metrics import PQCounts


def score_with_sizes(labels: np.ndarray, views: list[list[np.ndarray]]) -> tuple[PQCounts, dict]:
    flat = labels.ravel()
    areas = np.bincount(flat)
    ids = np.flatnonzero(areas[1:]) + 1
    counts = PQCounts()
    sizes = defaultdict(lambda: {'objects': 0, 'matched': 0})
    for view in views:
        matched_predictions = set()
        matched_gt = 0
        for pixels in view:
            overlap = np.bincount(flat[pixels], minlength=len(areas))
            iou = overlap / np.maximum(len(pixels) + areas - overlap, 1)
            hits = np.flatnonzero(iou[1:] > 0.5) + 1
            counts.tp += len(hits)
            counts.iou_sum += float(iou[hits].sum())
            matched = int(len(hits) > 0)
            matched_gt += matched
            matched_predictions.update(hits.tolist())
            size = ('below_400' if len(pixels) < 400 else '400_999' if len(pixels) < 1000
                    else '1000_2499' if len(pixels) < 2500 else '2500_9999' if len(pixels) < 10000 else '10000_plus')
            sizes[size]['objects'] += 1
            sizes[size]['matched'] += matched
        counts.fp += len(ids) - len(matched_predictions)
        counts.fn += len(view) - matched_gt
    return counts, sizes


def as_dict(counts: PQCounts) -> dict:
    return {**vars(counts), 'pq': counts.pq}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', default='data/kagglehub/competitions/filament-segmentation-2026')
    parser.add_argument('--settings', type=Path, default=Path('results/neural.json'))
    parser.add_argument('--labels-dir', type=Path, help='Evaluate existing labels without loading the model')
    parser.add_argument('--save-labels', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding='utf-8'))['settings']
    data = find_data_root(args.data)
    grouped = group_views(data.load_annotations())
    checkpoint_info = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    training_names, validation = checkpoint_split(grouped, checkpoint_info, validation_filenames(grouped))
    annotations_digest = hashlib.sha256(data.annotations_path.read_bytes()).hexdigest()
    if checkpoint_info.get('annotations_sha256', annotations_digest) != annotations_digest:
        raise ValueError('Checkpoint annotations differ from evaluation data')
    feature_recipe = checkpoint_info.get('feature_recipe', 'contrast')
    del checkpoint_info
    names = sorted(validation)[:args.limit]
    training_dates = {name[:8] for name in training_names}
    separate_dates = [name for name in names if name[:8] not in training_dates]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tile = (None, None) if args.labels_dir else load_model(args.checkpoint, device)
    if args.save_labels:
        args.save_labels.mkdir(parents=True, exist_ok=True)
    totals = {r: PQCounts() for r in (0, 1)}
    separate_date_totals = {r: PQCounts() for r in (0, 1)}
    sites = {r: defaultdict(PQCounts) for r in (0, 1)}
    sizes = {r: defaultdict(lambda: {'objects': 0, 'matched': 0}) for r in (0, 1)}
    rows = {r: [] for r in (0, 1)}
    started = time.monotonic()
    for index, name in enumerate(names, 1):
        if args.labels_dir:
            labels = cv2.imread(str(args.labels_dir / f'{Path(name).stem}.png'), cv2.IMREAD_UNCHANGED)
            if labels is None:
                raise FileNotFoundError(name)
        else:
            image = read_grayscale(data.train_dir / name)
            probability = predict_probability(image, model, device, tile, batch_size=args.batch_size)
            labels = hysteresis_to_labels(image, probability, settings['threshold'], settings['grow_threshold'],
                                          settings['min_area'], settings['max_area'], settings['limb_margin']).astype(np.uint16)
        if args.save_labels:
            if not cv2.imwrite(str(args.save_labels / f'{Path(name).stem}.png'), labels):
                raise OSError(name)
        decoded = [[np.flatnonzero(annotation_mask(a, labels.shape)) for a in view] for view in grouped[name]]
        for radius in (0, 1):
            counts, bins = score_with_sizes(grow_boundaries(labels, -radius), decoded)
            totals[radius] += counts
            if name in separate_dates:
                separate_date_totals[radius] += counts
            sites[radius][Path(name).stem[-2:]] += counts
            rows[radius].append({'filename': name, **as_dict(counts)})
            for key, value in bins.items():
                for metric in ('objects', 'matched'):
                    sizes[radius][key][metric] += value[metric]
        if index % 10 == 0 or index == len(names):
            print(f'{index}/{len(names)} | raw={totals[0].pq:.4f} eroded={totals[1].pq:.4f}', flush=True)
    payload = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               'settings': settings, 'images': len(names), 'inference_batch_size': args.batch_size,
               'feature_recipe': feature_recipe, 'training_filenames': sorted(training_names),
               'validation_filenames': sorted(validation), 'annotations_sha256': annotations_digest,
               'seconds': round(time.monotonic() - started, 1),
               'candidates': [{'erosion_radius': r, **as_dict(totals[r]), 'by_size': dict(sizes[r]),
                               'by_observatory': {k: as_dict(v) for k, v in sites[r].items()}, 'observations': rows[r],
                               'dates_absent_from_training': {'images': len(separate_dates), **as_dict(separate_date_totals[r])}}
                              for r in (0, 1)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'scores': [as_dict(totals[r]) for r in (0, 1)]}, indent=2))


if __name__ == '__main__':
    main()
