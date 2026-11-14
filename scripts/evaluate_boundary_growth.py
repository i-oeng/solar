"""Evaluate a small boundary-growth ablation using cached validation instances."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from solar_filament.data import annotation_mask, find_data_root, group_views, validation_filenames
from solar_filament.metrics import PQCounts


ROOT = Path(__file__).resolve().parents[1]


def grow_boundaries(labels: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return labels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(radius) + 1,) * 2)
    if radius < 0:
        eroded = cv2.erode(labels, kernel)
        # A pixel survives only if its whole neighborhood belongs to this ID.
        same_max = cv2.dilate(labels, kernel) == labels
        return np.where((eroded == labels) & same_max, labels, 0).astype(labels.dtype)
    grown = cv2.dilate(labels, kernel)
    # Preserve every original pixel. Higher instance ID resolves new-pixel ties.
    grown[labels > 0] = labels[labels > 0]
    return grown


def score_decoded(labels: np.ndarray, views: list[list[np.ndarray]]) -> PQCounts:
    """Organizer hit-based metric with GT pixel indices decoded once per image."""
    flat = labels.ravel()
    areas = np.bincount(flat)
    prediction_ids = np.flatnonzero(areas[1:]) + 1
    total = PQCounts()
    for view in views:
        hit_predictions = set()
        matched_gt = 0
        for pixels in view:
            intersections = np.bincount(flat[pixels], minlength=len(areas))
            ious = intersections / np.maximum(len(pixels) + areas - intersections, 1)
            hits = np.flatnonzero(ious[1:] > 0.5) + 1
            total.tp += len(hits)
            total.iou_sum += float(ious[hits].sum())
            matched_gt += int(len(hits) > 0)
            hit_predictions.update(hits.tolist())
        total.fp += len(prediction_ids) - len(hit_predictions)
        total.fn += len(view) - matched_gt
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--radii', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--output', type=Path, default=Path('results/epoch12_boundary_growth.json'))
    args = parser.parse_args()
    record = json.loads((ROOT / 'results/neural.json').read_text(encoding='utf-8'))
    data = find_data_root(ROOT / 'data/kagglehub/competitions/filament-segmentation-2026')
    grouped = group_views(data.load_annotations())
    names = sorted(validation_filenames(grouped))[:args.limit]
    totals = {r: PQCounts() for r in args.radii}
    sites = {r: defaultdict(PQCounts) for r in args.radii}
    observations = {r: [] for r in args.radii}
    for index, name in enumerate(names, 1):
        labels = cv2.imread(str(ROOT / f"outputs/epoch{record['train_epochs']}_validation_masks" / (Path(name).stem + '.png')), cv2.IMREAD_UNCHANGED)
        if labels is None:
            raise FileNotFoundError(name)
        views = [[np.flatnonzero(annotation_mask(a, labels.shape)) for a in view] for view in grouped[name]]
        for radius in args.radii:
            counts = score_decoded(grow_boundaries(labels, radius), views)
            totals[radius] += counts
            sites[radius][Path(name).stem[-2:]] += counts
            observations[radius].append({'filename': name, **vars(counts), 'pq': counts.pq})
        if index % 10 == 0 or index == len(names):
            print(f'{index}/{len(names)} | ' + ' | '.join(f'r={r}: {v.pq:.4f}' for r, v in totals.items()), flush=True)
    if len(names) == record['validation_images'] and 0 in totals:
        baseline = totals[0]
        assert (baseline.tp, baseline.fp, baseline.fn) == (record['tp'], record['fp'], record['fn'])
        assert abs(baseline.pq - record['pq']) < 1e-12
    result = {'checkpoint': record['checkpoint'], 'images': len(names),
              'note': 'Development ablation. Negative radius erodes each instance. Positive radius retains original pixels and resolves grown-pixel collisions by higher ID.',
              'candidates': [{'radius': r, **vars(v), 'pq': v.pq,
                              'by_observatory': {site: {**vars(c), 'pq': c.pq} for site, c in sites[r].items()},
                              'observations': observations[r]} for r, v in totals.items()]}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
