"""Blend uint16 probability map directories for --from-maps submission runs.

Each ``--group`` is a list of directories averaged with equal weights (for example the
three fold models of one architecture); the group means are then averaged with equal
weights, matching the ``a+b`` blends scored on development maps, where each side is one
map per image. Directories may hold the PNGs directly or under ``probability/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def png_dir(path: Path) -> Path:
    return path / 'probability' if (path / 'probability').is_dir() else path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', nargs='+', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--common', action='store_true', help='blend only the images present in every directory')
    args = parser.parse_args()
    groups = [[png_dir(d) for d in group] for group in args.group]
    stems = sorted(p.stem for p in groups[0][0].glob('*.png'))
    if args.common:
        for group in groups:
            for directory in group:
                stems = sorted(set(stems) & {p.stem for p in directory.glob('*.png')})
    for group in groups:
        for directory in group:
            have = sorted(p.stem for p in directory.glob('*.png'))
            if args.common:
                have = [stem for stem in have if stem in set(stems)]
            if have != stems:
                raise ValueError(f'{directory} has {len(have)} maps, expected the same {len(stems)} as {groups[0][0]}')
    args.output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for stem in stems:
        total = np.zeros((2048, 2048), np.float64)
        for group in groups:
            part = np.zeros((2048, 2048), np.float64)
            for directory in group:
                p = cv2.imread(str(directory / f'{stem}.png'), cv2.IMREAD_UNCHANGED)
                if p is None or p.dtype != np.uint16 or p.shape != (2048, 2048):
                    raise ValueError(f'Bad map {directory / stem}.png')
                part += p / 65535.0
            total += part / len(group)
        out = np.round(np.clip(total / len(groups), 0, 1) * 65535).astype(np.uint16)
        cv2.imwrite(str(args.output / f'{stem}.png'), out)
        digest.update(out.tobytes())
    provenance = {'groups': [[str(d) for d in g] for g in groups], 'images': len(stems),
                  'weights': 'equal within each group, then equal across groups',
                  'output_sha256': digest.hexdigest()}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
