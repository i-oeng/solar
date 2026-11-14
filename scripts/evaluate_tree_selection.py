"""Leave-one-fold-out evaluation of multi-threshold candidates with exact antichain selection.

For every development image, candidates are cut from the out-of-fold map at each grow
threshold in ``--levels`` (``solar_filament.candidate_tree``). The LightGBM scorer (same
parameters, target and weights as ``evaluate_rescoring.py``, plus four tree features) is
trained on the other two folds' nodes and predicts each node's mean matched IoU per view.
The kept set is the antichain maximising the sum of (prediction - t); t comes from the
usual date-grouped inner cross-validation on the training folds, now scored with the
antichain selection.

Fixed before scoring: the level sets below, the map variant, the scorer recipe and the
threshold grid. The single level (0.5,) with the standard features is the reference; it
must reproduce the 0.4397 blend result. Paired date bootstrap intervals compare every
level set with the reference. The oracle (true IoU, Dinkelbach iteration) is reported as
the ceiling of each level set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

from confirm_frequency_candidate import date_paired
from evaluate_global_folds import expand
from evaluate_global_gate import load_map
from evaluate_rescoring import CATEGORICAL, PARAMS, THRESHOLDS, image_rows, pooled
from solar_filament.candidate_tree import CONTEXT_FEATURES, TREE_FEATURES, add_context, build_tree, select_antichain
from solar_filament.data import annotation_mask, find_data_root, group_views, read_grayscale
from solar_filament.rescoring import FEATURES, match_candidates

ALL_LEVELS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
LEVEL_SETS = [(0.5,), (0.4, 0.5, 0.6), (0.3, 0.4, 0.5, 0.6, 0.7), ALL_LEVELS]
_WORKER = {}


def _init(spec, data_root, settings):
    data = find_data_root(data_root)
    _WORKER.update(spec=spec, data=data, grouped=group_views(data.load_annotations()), settings=settings)


def _image_nodes(name: str) -> tuple[pd.DataFrame, dict]:
    data, grouped, settings = _WORKER['data'], _WORKER['grouped'], _WORKER['settings']
    image = read_grayscale(data.train_dir / name)
    probability = load_map(_WORKER['spec'], Path(name).stem)
    tree = build_tree(image, probability, ALL_LEVELS, settings, observatory=Path(name).stem[-2:])
    views = [[np.flatnonzero(annotation_mask(a, image.shape)) for a in view] for view in grouped[name]]
    columns = {k: [] for k in ('iou_sum', 'hits', 'matched_views')}
    for depth in range(len(ALL_LEVELS)):
        m = match_candidates(tree['labels'][depth], views)
        for k in columns:
            columns[k].append(m[k])
    frame = pd.DataFrame(tree['features'], columns=list(FEATURES) + list(TREE_FEATURES))
    frame['filename'], frame['depth'], frame['parent_local'] = name, tree['level_index'], tree['parent']
    for k, v in columns.items():
        frame[k] = np.concatenate(v)
    frame['views'] = m['views']
    return frame, {'ground_truth': m['ground_truth'], 'views': m['views']}


def fold_table(spec: str, fold: int, data_root: str, settings: dict, cache: Path, workers: int):
    key = hashlib.sha256(f'tree|{fold}|{spec}|{ALL_LEVELS}'.encode()).hexdigest()[:16]
    path, meta = cache / f'{key}.csv.gz', cache / f'{key}.json'
    if path.exists() and meta.exists():
        return pd.read_csv(path, dtype={'filename': str}), json.loads(meta.read_text())
    names = sorted(json.loads(Path(f'results/annotation_round_folds/fold_{fold}.json').read_text())['validation_filenames'])
    with Pool(workers, initializer=_init, initargs=(spec, data_root, settings)) as pool:
        results = pool.map(_image_nodes, names, chunksize=2)
    table = pd.concat([r[0] for r in results], ignore_index=True)
    images = {name: r[1] for name, r in zip(names, results)}
    table.to_csv(path, index=False)
    meta.write_text(json.dumps(images))
    print(f'fold {fold}: {len(table)} nodes over {len(names)} images', flush=True)
    return table, images


def restrict(table: pd.DataFrame, levels: tuple[float, ...]) -> pd.DataFrame:
    """Nodes at the chosen levels, with global parent indices to the nearest kept ancestor."""
    table = table.reset_index(drop=True)
    start = table.groupby('filename', sort=False).cumcount().to_numpy()
    first = (np.arange(len(table)) - start)                        # row of the image's first node
    parent = np.where(table['parent_local'] >= 0, first + table['parent_local'].to_numpy(), -1)
    level = table['level'].to_numpy()
    kept = np.isin(np.round(level, 3), np.round(levels, 3))
    while True:  # climb past dropped levels
        climbing = (parent >= 0) & ~kept[np.maximum(parent, 0)]
        if not climbing.any():
            break
        parent[climbing] = parent[parent[climbing]]
    rows = np.flatnonzero(kept)
    new_index = np.full(len(table), -1)
    new_index[rows] = np.arange(len(rows))
    out = table.iloc[rows].copy().reset_index(drop=True)
    out['parent'] = np.where(parent[rows] >= 0, new_index[np.maximum(parent[rows], 0)], -1)
    out['depth'] = np.searchsorted(np.round(sorted(levels), 3), np.round(out['level'].to_numpy(), 3))
    out['date'] = out['filename'].str[:8]
    out['target'] = out['iou_sum'] / out['views']
    return out


def features_for(levels: tuple[float, ...], context: bool = False) -> list[str]:
    if len(levels) == 1:
        return list(FEATURES)
    return list(FEATURES) + list(TREE_FEATURES) + (list(CONTEXT_FEATURES) if context else [])


def fit(table: pd.DataFrame, features: list[str]) -> lightgbm.LGBMRegressor:
    model = lightgbm.LGBMRegressor(**PARAMS)
    model.fit(table[features], table['target'], sample_weight=table['views'], categorical_feature=CATEGORICAL)
    return model


def select(table: pd.DataFrame, score: np.ndarray, t: float) -> np.ndarray:
    return select_antichain(score - t, table['parent'].to_numpy(), table['depth'].to_numpy())


def counts_pq(table: pd.DataFrame, keep: np.ndarray, ground_truth: int) -> float:
    tp = table['hits'].to_numpy()[keep].sum()
    fp = (table['views'] - table['matched_views']).to_numpy()[keep].sum()
    denominator = tp + 0.5 * (fp + ground_truth - tp)
    return float(table['iou_sum'].to_numpy()[keep].sum() / denominator) if denominator else 0.0


def choose_threshold(table: pd.DataFrame, ground_truth: int, features: list[str]) -> tuple[float, dict]:
    dates = np.array(sorted(table['date'].unique()))
    order = np.argsort([hashlib.sha256(f'rescoring-inner:2026:{d}'.encode()).hexdigest() for d in dates])
    groups = {d: i % 5 for i, d in enumerate(dates[order])}
    inner = table['date'].map(groups).to_numpy()
    predictions = np.zeros(len(table))
    for k in range(5):
        model = fit(table[inner != k], features)
        predictions[inner == k] = model.predict(table.loc[inner == k, features])
    curve = {float(t): counts_pq(table, select(table, predictions, t), ground_truth) for t in THRESHOLDS}
    best = max(curve, key=lambda t: (curve[t], -abs(t - 0.2)))
    return best, curve


def oracle(table: pd.DataFrame, ground_truth: int) -> float:
    target = table['target'].to_numpy()
    pq = 0.4
    for _ in range(30):
        pq = counts_pq(table, select(table, target, pq / 2), ground_truth)
    return pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='outputs/{rk}+outputs/global_f{k}_1536')
    parser.add_argument('--data', default='data/kagglehub/competitions/filament-segmentation-2026')
    parser.add_argument('--settings', type=Path, default=Path('results/neural_resnet34_epoch30_eroded.json'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/tree_tables'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context', action='store_true', help='add parent, child and same level context features')
    parser.add_argument('--export', type=Path, help='fit the scorer of the best level set on all folds and save it')
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())['settings']
    args.cache.mkdir(parents=True, exist_ok=True)
    tables = {k: fold_table(expand(args.variant, k), k, args.data, settings, args.cache, args.workers) for k in range(3)}
    report, all_rows = {'variant': args.variant, 'levels_built': ALL_LEVELS}, {}
    for levels in LEVEL_SETS:
        name = ','.join(f'{v:g}' for v in levels)
        features = features_for(levels, args.context)
        views = {k: restrict(tables[k][0], levels) for k in range(3)}
        if args.context and len(levels) > 1:
            views = {k: add_context(v) for k, v in views.items()}
        rows, per_fold = [], {}
        for held in range(3):
            # parent indices are per fold table; shift them when concatenating
            offsets = np.cumsum([0] + [len(views[k]) for k in range(3) if k != held])[:-1]
            parts = []
            for offset, k in zip(offsets, [k for k in range(3) if k != held]):
                part = views[k].copy()
                part['parent'] = np.where(part['parent'] >= 0, part['parent'] + offset, -1)
                parts.append(part)
            train = pd.concat(parts, ignore_index=True)
            truth = sum(v['ground_truth'] for k in range(3) if k != held for v in tables[k][1].values())
            threshold, _ = choose_threshold(train, truth, features)
            model = fit(train, features)
            test = views[held]
            keep = select(test, model.predict(test[features]), threshold)
            fold_rows = image_rows(test, keep, tables[held][1])
            per_fold[held] = {'threshold': threshold, **pooled(fold_rows), 'kept': int(keep.sum()),
                              'nodes': len(test),
                              'oracle_pq': oracle(test, sum(v['ground_truth'] for v in tables[held][1].values()))}
            rows += fold_rows
        all_rows[name] = rows
        report[name] = {'pooled': pooled(rows), 'folds': per_fold}
        print(f'{name}: {report[name]["pooled"]["pq"]:.4f}  ' + '  '.join(
            f'f{k} {v["pq"]:.4f}@{v["threshold"]:.2f} oracle {v["oracle_pq"]:.3f}' for k, v in per_fold.items()), flush=True)
    reference = ','.join(f'{v:g}' for v in LEVEL_SETS[0])
    for levels in LEVEL_SETS[1:]:
        name = ','.join(f'{v:g}' for v in levels)
        report[name]['vs_reference'] = d = date_paired(all_rows[reference], all_rows[name])
        print(f'{name} vs {reference}: {d["delta_pq"]:+.4f} [{d["paired_95_interval"][0]:+.4f}, '
              f'{d["paired_95_interval"][1]:+.4f}]', flush=True)
    if args.export:
        name = max((n for n in all_rows), key=lambda n: report[n]['pooled']['pq'])
        levels = tuple(float(v) for v in name.split(','))
        features = features_for(levels, args.context)
        parts, offset = [], 0
        for k in range(3):
            part = restrict(tables[k][0], levels)
            if args.context and len(levels) > 1:
                part = add_context(part)
            part['parent'] = np.where(part['parent'] >= 0, part['parent'] + offset, -1)
            offset += len(part)
            parts.append(part)
        train = pd.concat(parts, ignore_index=True)
        truth = sum(v['ground_truth'] for k in range(3) for v in tables[k][1].values())
        threshold, curve = choose_threshold(train, truth, features)
        fit(train, features).booster_.save_model(str(args.export))
        report['export'] = {'levels': levels, 'features': features, 'path': str(args.export), 'threshold': threshold,
                            'inner_cv_pq': curve[threshold], 'nodes': len(train)}
        print(f'exported {args.export} levels {levels} threshold {threshold:.2f} inner CV {curve[threshold]:.4f}')
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
