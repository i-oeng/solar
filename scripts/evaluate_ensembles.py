"""Frozen E1/E2 evaluation with calibration-only choice and paired intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from evaluate_checkpoint import as_dict, score_with_sizes
from solar_filament.data import annotation_mask, find_data_root, group_views, read_grayscale, validation_filenames
from solar_filament.ensemble import average_probabilities
from solar_filament.inference import hysteresis_to_labels
from solar_filament.metrics import PQCounts
from solar_filament.refine import erode_labels


def summarize(rows: list[dict]) -> dict:
    total = PQCounts()
    sizes = defaultdict(lambda: {'objects': 0, 'matched': 0})
    sites = defaultdict(PQCounts)
    for row in rows:
        counts = PQCounts(**{k: row[k] for k in ('iou_sum', 'tp', 'fp', 'fn')})
        total += counts
        sites[Path(row['filename']).stem[-2:]] += counts
        for key, value in row['by_size'].items():
            for metric in ('objects', 'matched'):
                sizes[key][metric] += value[metric]
    for value in sizes.values():
        value['recall'] = value['matched'] / value['objects'] if value['objects'] else 0
    return {'images': len(rows), **as_dict(total), 'by_size': dict(sizes),
            'by_observatory': {k: as_dict(v) for k, v in sites.items()}}


def paired(before: list[dict], after: list[dict]) -> dict:
    before = sorted(before, key=lambda r: r['filename'])
    after = sorted(after, key=lambda r: r['filename'])
    if [r['filename'] for r in before] != [r['filename'] for r in after]:
        raise ValueError('Paired observations differ')
    fields = ('iou_sum', 'tp', 'fp', 'fn')
    left = np.array([[r[k] for k in fields] for r in before])
    right = np.array([[r[k] for k in fields] for r in after])
    indices = np.random.default_rng(2026).integers(len(before), size=(10000, len(before)))
    def pq(values):
        denominator = values[..., 1] + 0.5 * (values[..., 2] + values[..., 3])
        return np.divide(values[..., 0], denominator, out=np.zeros_like(denominator), where=denominator > 0)
    differences = pq(right[indices].sum(1)) - pq(left[indices].sum(1))
    return {'delta_pq': float(pq(right.sum(0)) - pq(left.sum(0))),
            'paired_95_interval': np.quantile(differences, [0.025, 0.975]).tolist(),
            'improved_observations': int((pq(right) > pq(left) + 1e-12).sum()),
            'worse_observations': int((pq(right) < pq(left) - 1e-12).sum()),
            'replicates': 10000, 'seed': 2026, 'unit': 'physical observation',
            'note': 'Descriptive development interval; no selection or temporal-dependence correction.'}


def load_cache(root: Path, mode: str, name: str, metadata: dict) -> np.ndarray:
    path = root / mode / f'{Path(name).stem}.npy'
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata[mode]['sha256']:
        raise ValueError(f'Cache checksum mismatch: {path}')
    array = np.load(path, allow_pickle=False)
    if array.shape != (2048, 2048) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f'Malformed cache: {path}')
    if array.min() < 0 or array.max() > 1:
        raise ValueError(f'Invalid probabilities: {path}')
    return array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch30', type=Path, default=Path('outputs/ensemble_epoch30'))
    parser.add_argument('--epoch20', type=Path, default=Path('outputs/ensemble_epoch20'))
    parser.add_argument('--output', type=Path, default=Path('results/ensemble_frozen.json'))
    parser.add_argument('--masks', type=Path, default=Path('outputs/ensemble_validation_masks'))
    parser.add_argument('--e1-only', action='store_true')
    parser.add_argument('--wait-for-cache', type=float, default=0,
                        help='Maximum seconds to wait for each observation from a running cache producer')
    parser.add_argument('--d4', action='store_true')
    parser.add_argument('--combine-weight', type=float, help='Previously selected epoch30 weight; requires matching augmented maps for both models')
    parser.add_argument('--combine-tta', choices=('flip4', 'd4'), default='flip4')
    parser.add_argument('--data', default='data/kagglehub/competitions/filament-segmentation-2026')
    args = parser.parse_args()
    reference = json.loads(Path('results/neural_resnet34_epoch30_eroded.json').read_text())
    data = find_data_root(args.data)
    grouped = group_views(data.load_annotations())
    names = sorted(validation_filenames(grouped))
    calibration = set(reference['calibration_filenames'])
    if len(calibration) != 40 or not calibration.issubset(names):
        raise ValueError('Fixed calibration subset is invalid')
    training_dates = {n[:8] for n in set(grouped) - set(names)}
    roots = {'epoch30': args.epoch30}
    if not args.e1_only:
        roots['epoch20'] = args.epoch20
    manifests = {key: json.loads((root / 'manifest.json').read_text()) for key, root in roots.items()}
    for key, manifest in manifests.items():
        if manifest['filenames'] != names or manifest['annotations_sha256'] != hashlib.sha256(data.annotations_path.read_bytes()).hexdigest():
            raise ValueError(f'{key} cache split/annotations mismatch')
        if manifest['checkpoint_sha256'] != hashlib.sha256(Path(manifest['checkpoint']).read_bytes()).hexdigest():
            raise ValueError(f'{key} checkpoint changed')
        for field in ('input_sha256', 'tile_size', 'overlap', 'batch_size', 'dtype'):
            if manifest[field] != manifests['epoch30'][field]:
                raise ValueError(f'Inconsistent preprocessing: {field}')
        # Extra, unimported modules may be added while inference is running.
        for path, digest in manifests['epoch30']['source_sha256'].items():
            if manifest['source_sha256'].get(path) != digest:
                raise ValueError(f'Inference source changed between members: {path}')
    if manifests['epoch30']['checkpoint_sha256'] != reference['checkpoint_sha256']:
        raise ValueError('Epoch30 is not the frozen reference checkpoint')
    recipes = [('epoch30_none', 'none', 1.0), ('epoch30_flip4', 'flip4', 1.0)]
    if args.d4:
        recipes.append(('epoch30_d4', 'd4', 1.0))
    if not args.e1_only:
        recipes += [(f'ensemble_w{w:g}', 'none', w) for w in (0, 0.25, 0.5, 0.75)]
    if args.combine_weight is not None:
        if args.e1_only or not 0 < args.combine_weight < 1:
            raise ValueError('Combination requires both models and an interior weight')
        recipes.append((f'ensemble_{args.combine_tta}_w{args.combine_weight:g}', args.combine_tta, args.combine_weight))
    rows = {key: [] for key, _, _ in recipes}
    for key in rows:
        (args.masks / key).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    settings = reference['settings']
    for number, name in enumerate(names, 1):
        deadline = time.monotonic() + args.wait_for_cache
        required_modes = {'epoch30': {mode for _, mode, weight in recipes if weight > 0}}
        if not args.e1_only:
            required_modes['epoch20'] = {mode for _, mode, weight in recipes if weight < 1}
        while True:
            ready = True
            for model_key, modes in required_modes.items():
                metadata_path = roots[model_key] / f'{Path(name).stem}.json'
                try:
                    saved = json.loads(metadata_path.read_text())
                    ready &= all(mode in saved for mode in modes)
                except (FileNotFoundError, json.JSONDecodeError):
                    ready = False
            if ready:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Incomplete cache: {name}')
            time.sleep(1)
        image = read_grayscale(data.train_dir / name)
        if hashlib.sha256((data.train_dir / name).read_bytes()).hexdigest() != manifests['epoch30']['input_sha256'][name]:
            raise ValueError(f'Input changed: {name}')
        decoded = [[np.flatnonzero(annotation_mask(a, image.shape)) for a in view] for view in grouped[name]]
        metadata = {key: json.loads((root / f'{Path(name).stem}.json').read_text()) for key, root in roots.items()}
        cached = {}
        for key, mode, weight in recipes:
            sources = [('epoch30', weight)] if weight == 1 else [('epoch20', 1)] if weight == 0 else [('epoch30', weight), ('epoch20', 1 - weight)]
            for model_key, _ in sources:
                if (model_key, mode) not in cached:
                    cached[model_key, mode] = load_cache(roots[model_key], mode, name, metadata[model_key])
            t0 = time.perf_counter()
            probability = cached[sources[0][0], mode] if len(sources) == 1 else average_probabilities(cached['epoch30', mode], cached['epoch20', mode], weight)
            labels = erode_labels(hysteresis_to_labels(image, probability, settings['threshold'], settings['grow_threshold'],
                                                      settings['min_area'], settings['max_area'], settings['limb_margin']), 1)
            post_seconds = time.perf_counter() - t0
            counts, sizes = score_with_sizes(labels, decoded)
            measurements = [metadata[model_key][mode] for model_key, _ in sources]
            rows[key].append({'filename': name, **as_dict(counts), 'by_size': sizes,
                              'inference_seconds': sum(m['seconds'] for m in measurements),
                              'postprocess_seconds': post_seconds,
                              'peak_allocated_bytes': max(m['peak_allocated_bytes'] for m in measurements),
                              'peak_reserved_bytes': max(m['peak_reserved_bytes'] for m in measurements)})
            if not cv2.imwrite(str(args.masks / key / f'{Path(name).stem}.png'), labels.astype(np.uint16)):
                raise OSError(name)
        if number % 10 == 0 or number == len(names):
            print(f'{number}/{len(names)} ' + ' '.join(f'{k}={summarize(v)["pq"]:.4f}' for k, v in rows.items()), flush=True)
    candidates = []
    for key, mode, weight in recipes:
        observations = rows[key]
        candidates.append({'name': key, 'tta': mode, 'epoch30_weight': weight,
                           'all_development': summarize(observations),
                           'calibration': summarize([r for r in observations if r['filename'] in calibration]),
                           'remaining_development': summarize([r for r in observations if r['filename'] not in calibration]),
                           'dates_absent_from_training': summarize([r for r in observations if r['filename'][:8] not in training_dates]),
                           'paired_vs_baseline': paired(rows['epoch30_none'], observations),
                           'mean_full_image_seconds': float(np.mean([r['inference_seconds'] + r['postprocess_seconds'] for r in observations])),
                           'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in observations),
                           'peak_reserved_bytes': max(r['peak_reserved_bytes'] for r in observations),
                           'observations': observations})
    baseline = candidates[0]['all_development']
    if any(baseline[k] != reference[k] for k in ('tp', 'fp', 'fn')) or abs(baseline['pq'] - reference['pq']) > 1e-9:
        raise AssertionError(f'Baseline did not reproduce: {baseline}')
    selection = {'e1': max((c for c in candidates if c['epoch30_weight'] == 1), key=lambda c: c['calibration']['pq'])['name']}
    if not args.e1_only:
        selection['e2'] = max((c for c in candidates if c['tta'] == 'none'), key=lambda c: c['calibration']['pq'])['name']
    for candidate in candidates:
        if candidate['tta'] != 'none' and candidate['epoch30_weight'] < 1:
            candidate['paired_vs_e1_winner'] = paired(rows[selection['e1']], candidate['observations'])
            candidate['paired_vs_e2_winner'] = paired(rows[selection['e2']], candidate['observations'])
    payload = {'status': 'complete', 'settings': settings, 'erosion_radius': 1, 'cache_dtype': 'float32',
               'calibration_filenames': reference['calibration_filenames'], 'manifests': manifests,
               'selection': selection, 'selection_rule': 'Highest fixed-40 calibration PQ; remaining 101 and full 141 are development results.',
               'baseline_reproduced': True, 'candidates': candidates,
               'timing_note': 'Inference plus probability fusion/postprocessing; excludes cache I/O, model loading and annotation scoring. Sequential ensemble inference is the sum of measured member times.',
               'evaluation_seconds': time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'selection': selection, 'candidates': [
        {'name': c['name'], 'calibration_pq': c['calibration']['pq'],
         **{k: c['all_development'][k] for k in ('pq', 'tp', 'fp', 'fn')},
         'mean_full_image_seconds': c['mean_full_image_seconds']} for c in candidates]}, indent=2))


if __name__ == '__main__':
    main()
