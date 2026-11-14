"""Cache foreground pixels per annotated object for balanced crop sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import annotation_mask, find_data_root, group_views, validation_filenames
from .prepare import choose_view


def sample_object_point(points: np.ndarray, offsets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Choose an object uniformly, then a foreground pixel within that object."""
    if len(offsets) < 2 or len(points) == 0:
        raise ValueError('Object sampler needs at least one nonempty annotation')
    object_index = int(rng.integers(len(offsets) - 1))
    return points[rng.integers(offsets[object_index], offsets[object_index + 1])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/kagglehub/competitions/filament-segmentation-2026')
    parser.add_argument('--output', type=Path, default=Path('data/prepared/object_points'))
    parser.add_argument('--split-manifest', type=Path)
    args = parser.parse_args()
    data = find_data_root(args.data)
    grouped = group_views(data.load_annotations())
    validation = validation_filenames(grouped)
    if args.split_manifest:
        if args.output.resolve() == Path('data/prepared/object_points').resolve():
            raise ValueError('Use a separate --output directory to preserve the original split cache')
        split = json.loads(args.split_manifest.read_text())
        if split['annotations_sha256'] != hashlib.sha256(data.annotations_path.read_bytes()).hexdigest():
            raise ValueError('Split annotation checksum mismatch')
        validation = set(split['validation_filenames'])
        training = set(split['training_filenames'])
        if validation & training or validation | training != set(grouped):
            raise ValueError('Split must partition all physical observations')
        if {n[:8] for n in validation} & {n[:8] for n in training}:
            raise ValueError('Date split contains shared training/validation dates')
    names = sorted(set(grouped) - validation)
    args.output.mkdir(parents=True, exist_ok=True)
    objects = pixels = 0
    for index, name in enumerate(names, 1):
        chunks = []
        for annotation in choose_view(grouped[name]):
            y, x = np.nonzero(annotation_mask(annotation))
            if len(x):
                chunks.append(np.column_stack((x, y)).astype(np.uint16))
        points = np.concatenate(chunks) if chunks else np.empty((0, 2), dtype=np.uint16)
        offsets = np.cumsum([0] + [len(chunk) for chunk in chunks], dtype=np.int64)
        np.savez_compressed(args.output / f'{Path(name).stem}.npz', points=points, offsets=offsets)
        objects += len(chunks)
        pixels += len(points)
        if index % 50 == 0 or index == len(names):
            print(f'{index}/{len(names)} training observations | {objects} objects', flush=True)
    manifest = {'images': len(names), 'objects': objects, 'foreground_pixels_with_overlap': pixels,
                'filenames': names, 'view_selection': 'most instances, then summed annotated area',
                'annotations_sha256': hashlib.sha256(data.annotations_path.read_bytes()).hexdigest()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'filenames'}, indent=2))


if __name__ == '__main__':
    main()
