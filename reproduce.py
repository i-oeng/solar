"""Reproduce the submitted predictions from the released weights (no training).

    python reproduce.py --data DATA_DIR --weights weights --work work [--limit N]

DATA_DIR holds the competition files (``find_data_root`` accepts the folder that contains
``test/`` or the folder one level above). Steps, each of which skips work that already exists:

1. ResNet34 group: 12 U-Nets (crop level), 4 flip TTA, probability maps.
2. ConvNeXt group: 3 whole-disk U-Nets at 1536 px, 4 flip TTA, probability maps.
3. Average of the two group means, written as uint16 PNG.
4. Candidate forest at six thresholds, LightGBM scores, exact antichain selection, COCO RLE CSV.

With ``--limit N`` only the first N sorted test images get a row; use it to compare against
``submission_public_0.38.csv`` quickly (rows of those images must be identical).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEVELS = '0.3,0.4,0.5,0.6,0.7,0.8'
THRESHOLD = 0.27  # chosen by inner cross-validation on the training folds (results/tree_export_a.json)


def run(command: list[str]) -> None:
    print('+', ' '.join(str(c) for c in command), flush=True)
    subprocess.run([str(c) for c in command], check=True, cwd=ROOT,
                   env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(ROOT / 'scripts'), str(ROOT / 'src')])})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/kagglehub/competitions/filament-segmentation-2026')
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights')
    parser.add_argument('--work', type=Path, default=ROOT / 'work')
    parser.add_argument('--limit', type=int, default=None, help='only the first N sorted test images')
    args = parser.parse_args()
    weights, work = args.weights.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    resnets = sorted((weights / 'resnet34').glob('annotation_round_f*_s20*_*.pt'))
    convnexts = [weights / 'convnext' / f'f{k}_1536.pt' for k in range(3)]
    if len(resnets) != 12 or not all(p.exists() for p in convnexts):
        raise SystemExit(f'Expected 12 ResNet34 and 3 ConvNeXt weights under {weights}')
    limit = ['--limit', str(args.limit)] if args.limit else []

    if not (work / 'maps_resnet').exists():
        run([py, 'scripts/predict_ensemble_submission.py', '--data', args.data, '--members', *resnets, '--tta', 'flip4',
             '--maps-dir', work / 'maps_resnet', '--output', work / 'resnet_rule.csv', *limit])
    for k, path in enumerate(convnexts):
        if not (work / f'maps_convnext_f{k}').exists():
            run([py, 'scripts/train_global_unet.py', '--data', args.data, '--predict', '--test', '--size', '1536',
                 '--output', path, '--maps-dir', work / f'maps_convnext_f{k}'])
    if not (work / 'maps_blend').exists():
        run([py, 'scripts/blend_maps.py', '--group', work / 'maps_resnet', '--group',
             *[work / f'maps_convnext_f{k}' for k in range(3)], '--output', work / 'maps_blend', *(['--common'] if args.limit else [])])
    run([py, 'scripts/predict_ensemble_submission.py', '--data', args.data, '--from-maps', work / 'maps_blend',
         '--scorer', weights / 'rescorer_tree_a.txt', '--scorer-threshold', THRESHOLD, '--tree-levels', LEVELS,
         '--output', work / 'rescored.csv', *limit])
    if not args.limit:
        run([py, 'scripts/check_submission.py', work / 'rescored.csv', '--data', args.data])
    print(json.dumps({'output': str(work / 'rescored.csv')}))


if __name__ == '__main__':
    main()
