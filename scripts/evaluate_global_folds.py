"""Leave-one-fold-out scorer evaluation for map variants over all 707 development images.

A variant is a template with ``{k}`` for the fold, e.g. ``outputs/global_f{k}_1536`` or the
blend ``outputs/fold_ensemble_f{k}+outputs/global_f{k}_1536`` (fold 1 of the ResNet
ensemble lives in ``fold_ensemble_f1_x4``; write ``{rk}`` for that name). For each held-out
fold the LightGBM scorer is trained on the other two folds' candidates, its threshold
chosen by the usual date-grouped inner cross-validation, exactly as
``evaluate_rescoring_folds.py`` did for the 0.4222 result. The first variant is the
reference for paired date bootstrap intervals. ``--export`` fits the final scorer of the
LAST variant on all three folds and records its inner-CV threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from confirm_frequency_candidate import date_paired
from evaluate_global_gate import build
from evaluate_rescoring import FEATURES, choose_threshold, fit, image_rows, pooled
from solar_filament.data import find_data_root, group_views

RESNET = {0: 'fold_ensemble_f0', 1: 'fold_ensemble_f1_x4', 2: 'fold_ensemble_f2'}


def expand(template: str, fold: int) -> str:
    return template.replace('{rk}', RESNET[fold]).replace('{k}', str(fold))


def table_for(spec: str, fold: int, data, grouped, settings, cache: Path):
    key = hashlib.sha256(f'{fold}|{spec}'.encode()).hexdigest()[:16]
    path, meta = cache / f'{key}.csv.gz', cache / f'{key}.json'
    if path.exists() and meta.exists():
        table = pd.read_csv(path)
        table['date'] = table['filename'].str[:8]  # CSV round trips read dates as int
        return table, json.loads(meta.read_text())
    names = sorted(json.loads(Path(f'results/annotation_round_folds/fold_{fold}.json').read_text())['validation_filenames'])
    table, images = build(spec, names, data, grouped, settings)
    table.to_csv(path, index=False)
    meta.write_text(json.dumps(images))
    return table, images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('variants', nargs='+')
    parser.add_argument('--data', default='data/kagglehub/competitions/filament-segmentation-2026')
    parser.add_argument('--settings', type=Path, default=Path('results/neural_resnet34_epoch30_eroded.json'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/global_gate_tables'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export', type=Path, help='write the final scorer of the last variant here')
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())['settings']
    data = find_data_root(args.data)
    grouped = group_views(data.load_annotations())
    args.cache.mkdir(parents=True, exist_ok=True)
    report, all_rows = {}, {}
    for template in args.variants:
        tables = {k: table_for(expand(template, k), k, data, grouped, settings, args.cache) for k in range(3)}
        rows, per_fold = [], {}
        for held in range(3):
            train = pd.concat([tables[k][0] for k in range(3) if k != held], ignore_index=True)
            truth = sum(v['ground_truth'] for k in range(3) if k != held for v in tables[k][1].values())
            threshold, _ = choose_threshold(train, truth)
            model = fit(train)
            test, images = tables[held]
            keep = model.predict(test[list(FEATURES)]) > threshold
            fold_rows = image_rows(test, keep, images)
            per_fold[held] = {'threshold': threshold, **pooled(fold_rows)}
            rows += fold_rows
        all_rows[template] = rows
        report[template] = {'pooled': pooled(rows), 'folds': per_fold}
        print(f'{template}: learned {report[template]["pooled"]["pq"]:.4f}  '
              + '  '.join(f'f{k} {v["pq"]:.4f}@{v["threshold"]:.2f}' for k, v in per_fold.items()), flush=True)
    reference = args.variants[0]
    for template in args.variants[1:]:
        report[template]['vs_reference'] = d = date_paired(all_rows[reference], all_rows[template])
        print(f'{template} vs {reference}: {d["delta_pq"]:+.4f} '
              f'[{d["paired_95_interval"][0]:+.4f}, {d["paired_95_interval"][1]:+.4f}]')
    if args.export:
        template = args.variants[-1]
        tables = [table_for(expand(template, k), k, data, grouped, settings, args.cache) for k in range(3)]
        train = pd.concat([t[0] for t in tables], ignore_index=True)
        truth = sum(v['ground_truth'] for t in tables for v in t[1].values())
        threshold, curve = choose_threshold(train, truth)
        model = fit(train)
        model.booster_.save_model(str(args.export))
        report['export'] = {'variant': template, 'path': str(args.export), 'threshold': threshold,
                            'candidates': len(train), 'inner_cv_pq': curve[threshold]}
        print(f'exported {args.export} threshold {threshold:.2f} inner CV PQ {curve[threshold]:.4f}')
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
