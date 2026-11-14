"""Apply a measured boundary refinement to an existing COCO RLE submission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco_mask


def erode_instance(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius < 0:
        raise ValueError('Erosion radius must be nonnegative')
    if radius == 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    return cv2.erode(mask.astype(np.uint8), kernel)


def erode_labels(labels: np.ndarray, radius: int = 1) -> np.ndarray:
    """Erode disjoint instances while retaining IDs and the input dtype."""
    if radius < 0:
        raise ValueError('Erosion radius must be nonnegative')
    if radius == 0:
        return labels.copy()
    working = labels.astype(np.uint16 if labels.max() <= 65535 else np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    inside = (cv2.erode(working, kernel) == working) & (cv2.dilate(working, kernel) == working)
    return np.where(inside, labels, 0).astype(labels.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--radius', type=int, default=1)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError('Use a separate output to preserve the original submission')
    if args.radius < 0:
        raise ValueError('Erosion radius must be nonnegative')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with args.source.open(newline='', encoding='utf-8') as source, args.output.open('w', newline='', encoding='utf-8') as output:
        reader = csv.DictReader(source)
        if reader.fieldnames != ['filament_id', 'segmentation_rle']:
            raise ValueError('Unexpected source CSV columns')
        writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            rle = {'size': [2048, 2048], 'counts': row['segmentation_rle'].encode('ascii')}
            refined = erode_instance(coco_mask.decode(rle), args.radius)
            if not refined.any():
                dropped += 1
                continue
            encoded = coco_mask.encode(np.asfortranarray(refined))
            writer.writerow({'filament_id': row['filament_id'], 'segmentation_rle': encoded['counts'].decode('ascii')})
            kept += 1
    print(json.dumps({'source': str(args.source), 'output': str(args.output), 'radius': args.radius,
                      'masks': kept, 'dropped_empty': dropped,
                      'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
