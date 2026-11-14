"""Gate for the whole-disk ConvNeXt model: compare probability maps of one fold under the learned scorer.

Every variant is a directory of uint16 probability maps for the same held-out fold
(``<dir>/probability/<stem>.png``), or a blend ``a+b`` of two such directories (equal
weights). For each variant the script builds the candidate table with the accepted
settings, then scores the fold with a scorer cross-fitted inside the fold: dates are
split into five groups, each group is scored by a LightGBM model trained on the other
four, whose threshold is chosen by the usual inner cross-validation on those four. The
same procedure is applied to every variant, so the comparison is paired and fair, but the
scorer sees fewer candidates than the leave-one-fold-out numbers (about 0.8 of one fold
instead of two folds); absolute values are slightly lower.

Also reported: rule PQ, oracle selection PQ and the recall ceiling (share of annotated
filaments with some candidate above IoU 0.5). Deltas against the first variant come with
paired date bootstrap intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from confirm_frequency_candidate import date_paired
import evaluate_rescoring
from evaluate_rescoring import FEATURES, choose_threshold, fit, image_rows, pooled
from solar_filament.data import annotation_mask, find_data_root, group_views, read_grayscale
from solar_filament.rescoring import extract_candidates, match_candidates, subset_counts


def load_map(spec: str, stem: str) -> np.ndarray:
    parts = spec.split('+')
    total = None
    for part in parts:
        p = cv2.imread(str(Path(part) / 'probability' / f'{stem}.png'), cv2.IMREAD_UNCHANGED)
        if p is None or p.dtype != np.uint16:
            raise ValueError(f'Missing map {part} {stem}')
        p = p.astype(np.float32) / np.float32(65535)
        total = p if total is None else total + p
    return total / len(parts)


def stack_features(spec: str, stem: str, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Per-candidate statistics of each blended part and of its boundary map, if it has one."""
    flat = labels.ravel()
    n = int(flat.max())
    area = np.maximum(np.bincount(flat, minlength=n + 1)[1:], 1)
    mean = lambda values: np.bincount(flat, weights=values.ravel(), minlength=n + 1)[1:] / area
    out = {}
    for i, part in enumerate(spec.split('+')):
        p = cv2.imread(str(Path(part) / 'probability' / f'{stem}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535
        out[f'stack_p_mean_{i}'] = mean(p)
        out[f'stack_frac090_{i}'] = mean((p >= 0.9).astype(np.float32))
        boundary_path = Path(part) / 'boundary' / f'{stem}.png'
        if boundary_path.exists():
            b = cv2.resize(cv2.imread(str(boundary_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255, (2048, 2048),
                           interpolation=cv2.INTER_LINEAR)
            out[f'stack_bd_mean_{i}'] = mean(b)
            out[f'stack_bd_frac_{i}'] = mean((b >= 0.5).astype(np.float32))
    return out


def build(spec: str, names: list[str], data, grouped, settings, stack: bool = False) -> tuple[pd.DataFrame, dict]:
    frames, images = [], {}
    for number, name in enumerate(names, 1):
        image = read_grayscale(data.train_dir / name)
        probability = load_map(spec, Path(name).stem)
        views = [[np.flatnonzero(annotation_mask(a, image.shape)) for a in view] for view in grouped[name]]
        c = extract_candidates(image, probability, low=settings['grow_threshold'], high=settings['threshold'],
                               min_seed_area=settings['min_area'], max_area=settings['max_area'],
                               limb_margin=settings['limb_margin'], observatory=Path(name).stem[-2:])
        m = match_candidates(c.labels, views)
        n = len(c.accepted)
        images[name] = {'ground_truth': m['ground_truth'], 'views': m['views'],
                        'rule': subset_counts(m['iou_sum'], m['hits'], m['matched_views'], np.full(n, m['views']),
                                              c.accepted, m['ground_truth'])}
        frame = pd.DataFrame(c.features, columns=FEATURES)
        frame['filename'], frame['date'], frame['accepted'] = name, name[:8], c.accepted
        frame['iou_sum'], frame['hits'] = m['iou_sum'], m['hits']
        frame['matched_views'], frame['views'] = m['matched_views'], m['views']
        if stack:
            for key, values in stack_features(spec, Path(name).stem, c.labels).items():
                frame[key] = values
        frames.append(frame)
        if number % 60 == 0:
            print(f'  {spec}: {number}/{len(names)}', flush=True)
    table = pd.concat(frames, ignore_index=True)
    table['target'] = table['iou_sum'] / table['views']
    return table, images


def use_features(table: pd.DataFrame) -> list[str]:
    """The scorer's feature list: the standard 20 plus any stacked columns (observatory index unchanged)."""
    features = list(FEATURES) + sorted(c for c in table.columns if c.startswith('stack_'))
    evaluate_rescoring.FEATURES = tuple(features)
    return features


def crossfit(table: pd.DataFrame, images: dict) -> tuple[np.ndarray, list[float]]:
    features = use_features(table)
    dates = np.array(sorted({name[:8] for name in images}))
    order = np.argsort([hashlib.sha256(f'global-gate:2026:{d}'.encode()).hexdigest() for d in dates])
    groups = {d: i % 5 for i, d in enumerate(dates[order])}
    outer = table['filename'].str[:8].map(groups).to_numpy()  # CSV round trips read 'date' as int
    keep = np.zeros(len(table), bool)
    thresholds = []
    for k in range(5):
        train = table[outer != k]
        truth = sum(v['ground_truth'] for n, v in images.items() if groups[n[:8]] != k)
        threshold, _ = choose_threshold(train, truth)
        model = fit(train)
        keep[outer == k] = model.predict(table.loc[outer == k, features]) > threshold
        thresholds.append(threshold)
    return keep, thresholds


def oracle(table: pd.DataFrame, ground_truth: int) -> float:
    iou = table['iou_sum'].to_numpy()
    cost = 0.5 * (table['hits'] + table['views'] - table['matched_views']).to_numpy()
    lam = 0.4
    for _ in range(30):
        k = iou - lam * cost > 0
        lam = iou[k].sum() / (cost[k].sum() + 0.5 * ground_truth)
    return float(lam)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('variants', nargs='+', help='map directories or a+b blends; the first is the reference')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--data', default='data/kagglehub/competitions/filament-segmentation-2026')
    parser.add_argument('--settings', type=Path, default=Path('results/neural_resnet34_epoch30_eroded.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('outputs/global_gate_tables'))
    parser.add_argument('--stack', action='store_true', help='add per-part and boundary features to every variant')
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())['settings']
    data = find_data_root(args.data)
    grouped = group_views(data.load_annotations())
    names = sorted(json.loads(Path(f'results/annotation_round_folds/fold_{args.fold}.json').read_text())['validation_filenames'])
    args.cache.mkdir(parents=True, exist_ok=True)
    report, rows = {}, {}
    for label in args.variants:
        # 'a+b#stack' adds stacked features for that variant only
        spec, stack = label.removesuffix('#stack'), args.stack or label.endswith('#stack')
        key = hashlib.sha256(f'{args.fold}|{spec}{"|stack" if stack else ""}'.encode()).hexdigest()[:16]
        path, meta = args.cache / f'{key}.csv.gz', args.cache / f'{key}.json'
        if path.exists() and meta.exists():
            table, images = pd.read_csv(path), json.loads(meta.read_text())
        else:
            table, images = build(spec, names, data, grouped, settings, stack=stack)
            table.to_csv(path, index=False)
            meta.write_text(json.dumps(images))
        ground_truth = sum(v['ground_truth'] for v in images.values())
        keep, thresholds = crossfit(table, images)
        rows[label] = {'rule': image_rows(table, table['accepted'].to_numpy().astype(bool), images),
                      'learned': image_rows(table, keep, images)}
        spec = label
        recall = table['hits'].sum() / ground_truth
        report[spec] = {'candidates': len(table), 'rule': pooled(rows[spec]['rule']),
                        'learned': pooled(rows[spec]['learned']), 'thresholds': thresholds,
                        'oracle_pq': oracle(table, ground_truth), 'recall_ceiling': float(recall)}
        r = report[spec]
        print(f'{spec}: rule {r["rule"]["pq"]:.4f}  learned {r["learned"]["pq"]:.4f} '
              f'(TP {r["learned"]["tp"]} FP {r["learned"]["fp"]} FN {r["learned"]["fn"]})  '
              f'oracle {r["oracle_pq"]:.4f}  recall ceiling {recall:.3f}', flush=True)
    reference = args.variants[0]
    for spec in args.variants[1:]:
        report[spec]['learned_vs_reference'] = date_paired(rows[reference]['learned'], rows[spec]['learned'])
        d = report[spec]['learned_vs_reference']
        print(f'{spec} vs {reference}: learned delta {d["delta_pq"]:+.4f} '
              f'[{d["paired_95_interval"][0]:+.4f}, {d["paired_95_interval"][1]:+.4f}]')
    args.output.write_text(json.dumps({'fold': args.fold, 'settings': settings, 'variants': report}, indent=2))


if __name__ == '__main__':
    main()
