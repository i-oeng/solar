"""Versioned complete-view targets and matching object-centered crop caches."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .data import annotation_mask, find_data_root, group_views


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def view_index(seed, epoch, sample_index, count):
    if count < 1:
        raise ValueError('Each observation needs at least one annotation view')
    # This stream never consumes the crop/augmentation RNG.
    return int(np.random.default_rng([seed, epoch, sample_index, 113]).integers(count))


def prepare_views(data_path, output):
    data = find_data_root(data_path)
    grouped = group_views(data.load_annotations())
    expected = {'version': 1, 'annotations_sha256': digest(data.annotations_path),
                'filenames': sorted(grouped), 'view_counts': {n: len(v) for n, v in sorted(grouped.items())}}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(old[k] != v for k, v in expected.items()):
            raise ValueError('Annotation-view cache provenance differs')
        for relative, sha in old['files_sha256'].items():
            if digest(output / relative) != sha:
                raise ValueError(f'Annotation-view cache changed: {relative}')
        return old
    if output.exists() and any(output.iterdir()):
        raise ValueError('Incomplete cache exists; inspect before rebuilding in a new directory')
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    counts = {'views': 0, 'objects': 0, 'empty_views': 0}
    for number, name in enumerate(sorted(grouped), 1):
        directory = output / Path(name).stem
        directory.mkdir()
        for index, annotations in enumerate(grouped[name]):
            mask = np.zeros((2048, 2048), np.uint8)
            chunks = []
            for annotation in annotations:
                target = annotation_mask(annotation)
                mask[target] = 255
                y, x = np.nonzero(target)
                if len(x):
                    chunks.append(np.column_stack((x, y)).astype(np.uint16))
            points = np.concatenate(chunks) if chunks else np.empty((0, 2), np.uint16)
            offsets = np.cumsum([0] + [len(c) for c in chunks], dtype=np.int64)
            image_path, points_path = directory / f'{index}.png', directory / f'{index}.npz'
            if not cv2.imwrite(str(image_path), mask):
                raise OSError(image_path)
            np.savez_compressed(points_path, points=points, offsets=offsets)
            for path in (image_path, points_path):
                hashes[path.relative_to(output).as_posix()] = digest(path)
            counts['views'] += 1
            counts['objects'] += len(chunks)
            counts['empty_views'] += int(not chunks)
        if number % 50 == 0 or number == len(grouped):
            print(f'{number}/{len(grouped)} observations, {counts}', flush=True)
    payload = {**expected, **counts, 'files_sha256': hashes,
        'note': 'Cache includes all views. Training filenames alone control exposure; views never cross an observation split.'}
    manifest_path.write_text(json.dumps(payload, indent=2) + '\n')
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/kagglehub/competitions/filament-segmentation-2026')
    parser.add_argument('--output', type=Path, default=Path('data/prepared/annotation_views_v1'))
    args = parser.parse_args()
    result = prepare_views(args.data, args.output)
    print(json.dumps({k: result[k] for k in ('version', 'views', 'objects', 'empty_views')}))


if __name__ == '__main__':
    main()
