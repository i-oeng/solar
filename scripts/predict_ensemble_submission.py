"""Write a test submission from an equal-weight checkpoint ensemble with flip TTA.

Each member's probability map is the mean over the four aligned flips (or the
identity pass alone with --tta none); members are averaged with equal weights.
Postprocessing is the accepted seed rule with one-pixel erosion, or a learned
candidate scorer (--scorer, a LightGBM model file, with --scorer-threshold).
Only the supplied test images are read. With --from-maps, fused maps saved earlier
by --maps-dir (for example on another machine) replace model inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask

from solar_filament.data import find_data_root, read_grayscale
from solar_filament.ensemble import TTA_COUNTS, transform_image
from solar_filament.inference import hysteresis_to_labels, load_model, predict_probability
from solar_filament.refine import erode_labels
from solar_filament.rescoring import extract_candidates


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_labels(image, fused, booster, threshold, levels, settings, observatory):
    """Multi-threshold candidates, exact antichain selection with the tree scorer."""
    import pandas as pd
    from evaluate_tree_selection import ALL_LEVELS, restrict
    from solar_filament.candidate_tree import CONTEXT_FEATURES, TREE_FEATURES, add_context, build_tree, select_antichain
    from solar_filament.rescoring import FEATURES
    tree = build_tree(image, fused, ALL_LEVELS, settings, observatory=observatory)
    frame = pd.DataFrame(tree['features'], columns=list(FEATURES) + list(TREE_FEATURES))
    frame['filename'], frame['parent_local'], frame['orig_depth'] = 'x', tree['parent'], tree['level_index']
    frame['local'], frame['iou_sum'], frame['views'] = tree['local'], 0.0, 1
    nodes = restrict(frame, levels)
    if not len(nodes):
        return np.zeros(image.shape, np.uint16)
    columns = list(FEATURES) + list(TREE_FEATURES)
    if booster.num_feature() > len(columns):  # scorer trained with --context
        nodes, columns = add_context(nodes), columns + list(CONTEXT_FEATURES)
    score = booster.predict(nodes[columns])
    keep = select_antichain(score - threshold, nodes['parent'].to_numpy(), nodes['depth'].to_numpy())
    labels = np.zeros(image.shape, np.uint16)
    for number, (depth, local) in enumerate(nodes.loc[keep, ['orig_depth', 'local']].to_numpy(), 1):
        labels[tree['labels'][int(depth)] == int(local)] = number
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--members', type=Path, nargs='+')
    parser.add_argument('--from-maps', type=Path, help='Read fused uint16 PNG maps written by --maps-dir instead of running models')
    parser.add_argument('--tta', choices=('none', 'flip4'), default='flip4')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maps-dir', type=Path, help='Optionally keep fused maps as uint16 PNG (p * 65535)')
    parser.add_argument('--scorer', type=Path, help='LightGBM model text file from evaluate_rescoring.py')
    parser.add_argument('--scorer-threshold', type=float)
    parser.add_argument('--tree-levels', help='comma separated grow thresholds; the scorer must come from evaluate_tree_selection.py --export')
    parser.add_argument('--settings', type=Path, default=Path('results/neural_resnet34_epoch30_eroded.json'))
    parser.add_argument('--data', default='data/kagglehub/competitions/filament-segmentation-2026')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if (args.scorer is None) != (args.scorer_threshold is None):
        raise ValueError('--scorer and --scorer-threshold go together')
    if (args.members is None) == (args.from_maps is None):
        raise ValueError('Give exactly one of --members and --from-maps')
    settings = json.loads(args.settings.read_text())['settings']
    data = find_data_root(args.data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    members = {}
    for path in args.members or ():
        info = torch.load(path, map_location='cpu', weights_only=True)
        # Older checkpoints omit the recipe key; load_model treats that as contrast too.
        if info.get('epoch') != 30 or info.get('feature_recipe', 'contrast') != 'contrast':
            raise ValueError(f'Members must be completed 30-epoch contrast checkpoints: {path}')
        del info
        members[str(path)] = load_model(path, device)
    booster = None
    levels = tuple(float(v) for v in args.tree_levels.split(',')) if args.tree_levels else None
    if args.scorer:
        import lightgbm
        booster = lightgbm.Booster(model_file=str(args.scorer))
    if args.maps_dir:
        args.maps_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(data.test_dir.glob('*.jpeg'))[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    timings = []
    started = time.monotonic()
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['filament_id', 'segmentation_rle'])
        for number, path in enumerate(images, 1):
            image = read_grayscale(path)
            t0 = time.perf_counter()
            if args.from_maps:
                stored = cv2.imread(str(args.from_maps / f'{path.stem}.png'), cv2.IMREAD_UNCHANGED)
                if stored is None or stored.dtype != np.uint16 or stored.shape != image.shape:
                    raise ValueError(f'Missing or malformed map for {path.name}')
                fused = stored.astype(np.float32) / np.float32(65535)
            else:
                fused = np.zeros(image.shape, np.float32)
            for model, tile in members.values():
                total = np.zeros(image.shape, np.float32)
                for index in range(TTA_COUNTS[args.tta]):
                    probability = predict_probability(transform_image(image, index), model, device, tile,
                                                      batch_size=args.batch_size)
                    total += transform_image(probability, index, inverse=True)
                fused += total / np.float32(TTA_COUNTS[args.tta])
            if members:
                fused /= np.float32(len(members))
            if args.maps_dir:
                cv2.imwrite(str(args.maps_dir / f'{path.stem}.png'), np.round(np.clip(fused, 0, 1) * 65535).astype(np.uint16))
            if booster is not None and args.tree_levels:
                labels = tree_labels(image, fused, booster, args.scorer_threshold, levels, settings, path.stem[-2:])
            elif booster is None:
                labels = erode_labels(hysteresis_to_labels(image, fused, settings['threshold'], settings['grow_threshold'],
                                                           settings['min_area'], settings['max_area'],
                                                           settings['limb_margin']).astype(np.uint16), 1)
            else:
                candidates = extract_candidates(image, fused, low=settings['grow_threshold'], high=settings['threshold'],
                                                min_seed_area=settings['min_area'], max_area=settings['max_area'],
                                                limb_margin=settings['limb_margin'], observatory=path.stem[-2:])
                keep = booster.predict(candidates.features) > args.scorer_threshold if len(candidates.accepted) else np.zeros(0, bool)
                lookup = np.zeros(len(keep) + 1, np.uint16)
                lookup[1:][keep] = np.arange(1, int(keep.sum()) + 1)
                labels = lookup[candidates.labels]
            timings.append(time.perf_counter() - t0)
            for instance in range(1, int(labels.max()) + 1):
                mask = labels == instance
                if not mask.any():
                    continue
                rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
                writer.writerow([f'{path.stem}_{instance}', rle['counts'].decode('ascii')])
                rows += 1
            print(f'{number}/{len(images)} {path.stem}: {timings[-1]:.1f}s, {rows} masks', flush=True)
    provenance = {'output': str(args.output), 'output_sha256': digest(args.output), 'rows': rows, 'images': len(images),
                  'members': {name: digest(Path(name)) for name in members}, 'tta': args.tta if members else None,
                  'from_maps': {'directory': str(args.from_maps),
                                'provenance': json.loads(Path(args.from_maps, 'provenance.json').read_text())
                                if Path(args.from_maps, 'provenance.json').exists() else None,
                                'sha256_of_map_digests': hashlib.sha256(''.join(
                                    digest(args.from_maps / f'{p.stem}.png') for p in images).encode()).hexdigest()}
                  if args.from_maps else None,
                  'settings': settings,
                  'tree_levels': levels,
                  'postprocessing': 'learned candidate scorer' if booster else 'accepted seed rule, erosion 1',
                  'scorer': {'path': str(args.scorer), 'sha256': digest(args.scorer), 'threshold': args.scorer_threshold} if booster else None,
                  'mean_seconds_per_image': float(np.mean(timings)), 'total_seconds': time.monotonic() - started,
                  'device': str(device), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                  'torch': torch.__version__}
    args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({k: provenance[k] for k in ('rows', 'images', 'mean_seconds_per_image', 'output_sha256')}, indent=2))


if __name__ == '__main__':
    main()
