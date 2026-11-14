"""Cross-fitted learned candidate scoring versus the hand-written seed rule.

Inputs are the out-of-fold ensemble probability maps written by
evaluate_fold_ensembles.py for folds 0 and 1. No image in either fold was seen by
the models that produced its map. The scorer is trained on one fold's candidates
and evaluated on the other's, in both directions; its keep threshold is chosen from
date-grouped inner cross-validation predictions on the training fold only.

Fixed before scoring: LightGBM regressor, target = mean matched IoU per annotation
view, sample weight = number of views, the parameters below, and a threshold grid of
0.05 to 0.60 in steps of 0.01. No other variants are compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import lightgbm
import numpy as np
import pandas as pd

from confirm_frequency_candidate import date_paired
from solar_filament.data import annotation_mask, find_data_root, group_views, read_grayscale
from solar_filament.rescoring import FEATURES, extract_candidates, match_candidates, subset_counts

PARAMS = dict(objective='regression', n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=2026,
              deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)
THRESHOLDS = np.round(np.arange(0.05, 0.6001, 0.01), 2)
CATEGORICAL = [FEATURES.index('observatory')]


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_table(fold: int, data, grouped, settings) -> tuple[pd.DataFrame, dict]:
    root = Path(f'outputs/fold_ensemble_f{fold}')
    manifest = json.loads((root / 'manifest.json').read_text())
    result = json.loads(Path(f'results/fold_ensemble_f{fold}.json').read_text())
    if result['status'] != 'complete':
        raise ValueError(f'Fold {fold} ensemble evaluation is incomplete')
    recipe = manifest['probability_maps']['recipe']
    ensemble_rows = {r['filename']: r['rows'][recipe] for r in map(json.loads, (root / 'rows.jsonl').read_text().splitlines())}
    frames, images, agree = [], {}, 0
    for number, name in enumerate(manifest['filenames'], 1):
        image = read_grayscale(data.train_dir / name)
        probability = cv2.imread(str(root / 'probability' / f'{Path(name).stem}.png'), cv2.IMREAD_UNCHANGED)
        if probability is None or probability.dtype != np.uint16:
            raise ValueError(f'Missing probability map: {name}')
        probability = probability.astype(np.float32) / np.float32(65535)
        views = [[np.flatnonzero(annotation_mask(a, image.shape)) for a in view] for view in grouped[name]]
        candidates = extract_candidates(image, probability, low=settings['grow_threshold'], high=settings['threshold'],
                                        min_seed_area=settings['min_area'], max_area=settings['max_area'],
                                        limb_margin=settings['limb_margin'], observatory=Path(name).stem[-2:])
        matched = match_candidates(candidates.labels, views)
        n = len(candidates.accepted)
        rule = subset_counts(matched['iou_sum'], matched['hits'], matched['matched_views'], np.full(n, matched['views']),
                             candidates.accepted, matched['ground_truth'])
        # uint16 storage can move pixels lying within 1e-5 of a threshold; count exact agreement.
        agree += all(rule[k] == ensemble_rows[name][k] for k in ('tp', 'fp', 'fn'))
        images[name] = {'ground_truth': matched['ground_truth'], 'views': matched['views'], 'rule': rule}
        frame = pd.DataFrame(candidates.features, columns=FEATURES)
        frame['filename'] = name
        frame['date'] = name[:8]
        frame['accepted'] = candidates.accepted
        frame['iou_sum'] = matched['iou_sum']
        frame['hits'] = matched['hits']
        frame['matched_views'] = matched['matched_views']
        frame['views'] = matched['views']
        frames.append(frame)
        if number % 25 == 0 or number == len(manifest['filenames']):
            print(f'fold {fold}: {number}/{len(manifest["filenames"])} images, '
                  f'{sum(len(f) for f in frames)} candidates', flush=True)
    table = pd.concat(frames, ignore_index=True)
    table['target'] = table['iou_sum'] / table['views']
    info = {'recipe': recipe, 'images': len(images), 'candidates': len(table),
            'accepted_candidates': int(table['accepted'].sum()),
            'rule_matches_ensemble_rows': agree, 'members': manifest['members']}
    return table, {'images': images, 'info': info}


def image_rows(table: pd.DataFrame, keep: np.ndarray, images: dict) -> list[dict]:
    rows = []
    for name, meta in images.items():
        mask = (table['filename'] == name).to_numpy()
        part = table[mask]
        counts = subset_counts(part['iou_sum'].to_numpy(), part['hits'].to_numpy(), part['matched_views'].to_numpy(),
                               part['views'].to_numpy(), keep[mask], meta['ground_truth'])
        rows.append({'filename': name, **counts})
    return rows


def pooled(rows: list[dict]) -> dict:
    total = {k: sum(r[k] for r in rows) for k in ('iou_sum', 'tp', 'fp', 'fn')}
    denominator = total['tp'] + 0.5 * (total['fp'] + total['fn'])
    return {**total, 'pq': total['iou_sum'] / denominator if denominator else 0.0}


def fast_pq(table: pd.DataFrame, keep: np.ndarray, ground_truth: int) -> float:
    tp = table['hits'].to_numpy()[keep].sum()
    fp = (table['views'].to_numpy() - table['matched_views'].to_numpy())[keep].sum()
    denominator = tp + 0.5 * (fp + ground_truth - tp)
    return float(table['iou_sum'].to_numpy()[keep].sum() / denominator) if denominator else 0.0


def fit(table: pd.DataFrame) -> lightgbm.LGBMRegressor:
    model = lightgbm.LGBMRegressor(**PARAMS)
    model.fit(table[list(FEATURES)], table['target'], sample_weight=table['views'], categorical_feature=CATEGORICAL)
    return model


def choose_threshold(table: pd.DataFrame, ground_truth: int) -> tuple[float, dict]:
    dates = np.array(sorted(table['date'].unique()))
    order = np.argsort([hashlib.sha256(f'rescoring-inner:2026:{d}'.encode()).hexdigest() for d in dates])
    groups = {d: i % 5 for i, d in enumerate(dates[order])}
    inner = table['date'].map(groups).to_numpy()
    predictions = np.zeros(len(table))
    for k in range(5):
        model = fit(table[inner != k])
        predictions[inner == k] = model.predict(table.loc[inner == k, list(FEATURES)])
    curve = {float(t): fast_pq(table, predictions > t, ground_truth) for t in THRESHOLDS}
    best = max(curve, key=lambda t: (curve[t], -abs(t - 0.2)))
    return best, curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/kagglehub/competitions/filament-segmentation-2026')
    parser.add_argument('--settings', type=Path, default=Path('results/neural_resnet34_epoch30_eroded.json'))
    parser.add_argument('--output', type=Path, default=Path('results/rescoring_crossfit.json'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/rescoring'))
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())['settings']
    data = find_data_root(args.data)
    grouped = group_views(data.load_annotations())
    args.cache.mkdir(parents=True, exist_ok=True)
    tables, metas = {}, {}
    for fold in (0, 1):
        path = args.cache / f'candidates_f{fold}.csv.gz'
        meta_path = args.cache / f'candidates_f{fold}.json'
        if path.exists() and meta_path.exists():
            tables[fold], metas[fold] = pd.read_csv(path), json.loads(meta_path.read_text())
        else:
            tables[fold], metas[fold] = build_table(fold, data, grouped, settings)
            tables[fold].to_csv(path, index=False)
            meta_path.write_text(json.dumps(metas[fold], indent=2) + '\n')
    directions = []
    for train_fold, eval_fold in ((0, 1), (1, 0)):
        train, test = tables[train_fold], tables[eval_fold]
        train_gt = sum(m['ground_truth'] for m in metas[train_fold]['images'].values())
        threshold, curve = choose_threshold(train, train_gt)
        model = fit(train)
        predicted = model.predict(test[list(FEATURES)])
        rule_rows = image_rows(test, test['accepted'].to_numpy().astype(bool), metas[eval_fold]['images'])
        learned_rows = image_rows(test, predicted > threshold, metas[eval_fold]['images'])
        importance = dict(zip(FEATURES, model.booster_.feature_importance('gain').round(1).tolist()))
        directions.append({'train_fold': train_fold, 'eval_fold': eval_fold, 'threshold': threshold,
                           'inner_cv_pq_by_threshold': curve,
                           'train_rule_pq': fast_pq(train, train['accepted'].to_numpy().astype(bool), train_gt),
                           'eval_rule': pooled(rule_rows), 'eval_learned': pooled(learned_rows),
                           'kept_candidates': int((predicted > threshold).sum()),
                           'rule_candidates': int(test['accepted'].sum()), 'eval_candidates': len(test),
                           'paired_vs_rule': date_paired(rule_rows, learned_rows),
                           'feature_gain': dict(sorted(importance.items(), key=lambda kv: -kv[1]))})
        d = directions[-1]
        print(f'train f{train_fold} -> eval f{eval_fold}: tau={threshold:.2f} rule PQ {d["eval_rule"]["pq"]:.4f} '
              f'(TP {d["eval_rule"]["tp"]} FP {d["eval_rule"]["fp"]} FN {d["eval_rule"]["fn"]}) | learned PQ '
              f'{d["eval_learned"]["pq"]:.4f} (TP {d["eval_learned"]["tp"]} FP {d["eval_learned"]["fp"]} '
              f'FN {d["eval_learned"]["fn"]}) | delta {d["paired_vs_rule"]["delta_pq"]:+.4f} '
              f'{np.round(d["paired_vs_rule"]["paired_95_interval"], 4).tolist()}', flush=True)
    payload = {'status': 'complete', 'params': PARAMS, 'features': list(FEATURES), 'thresholds': THRESHOLDS.tolist(),
               'target': 'mean matched IoU per annotation view (keep raises PQ exactly when it exceeds PQ/2)',
               'sample_weight': 'annotation views of the image', 'tables': {f: metas[f]['info'] for f in metas},
               'directions': directions, 'script_sha256': digest(Path(__file__)),
               'rescoring_sha256': digest(Path('src/solar_filament/rescoring.py')),
               'note': ('Cross-fitted between folds 0 and 1; each map comes from models that never saw its dates. '
                        'Fold 0 maps average four members and fold 1 maps two, so each direction also tests transfer '
                        'across ensemble size. Development evidence only.')}
    args.output.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
