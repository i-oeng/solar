"""Competition data access and observation-level validation splits."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco_mask


@dataclass(frozen=True)
class CompetitionData:
    root: Path

    @property
    def train_dir(self) -> Path:
        return self.root / "train" / "train_images"

    @property
    def test_dir(self) -> Path:
        return self.root / "test" / "test_images"

    @property
    def annotations_path(self) -> Path:
        return self.root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    def load_annotations(self) -> dict:
        with self.annotations_path.open(encoding="utf-8") as handle:
            return json.load(handle)


def find_data_root(path: str | Path) -> CompetitionData:
    path = Path(path)
    candidates = [path, path / "MAGFiLO_1.0_Kaggle_2026"]
    for candidate in candidates:
        data = CompetitionData(candidate)
        if data.annotations_path.is_file() and data.test_dir.is_dir():
            return data
    raise FileNotFoundError(f"Competition data not found below {path}")


def group_views(metadata: dict) -> dict[str, list[list[dict]]]:
    """Group annotator views by physical observation filename."""
    images = {image["id"]: image["file_name"] for image in metadata["images"]}
    by_view: dict[str, list[dict]] = defaultdict(list)
    for annotation in metadata["annotations"]:
        by_view[annotation["image_id"]].append(annotation)
    grouped: dict[str, list[list[dict]]] = defaultdict(list)
    for view_id, filename in images.items():
        grouped[filename].append(by_view[view_id])
    return dict(grouped)


def validation_filenames(grouped: dict[str, list[list[dict]]], fraction: float = 0.2, seed: int = 2026) -> set[str]:
    """Choose entire observations so annotators never cross train and validation."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    ordered = sorted(
        grouped,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).digest(),
    )
    return set(ordered[:round(len(ordered) * fraction)])


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Could not read image: {path}")
    if image.shape != (2048, 2048):
        raise ValueError(f"Expected 2048x2048 image, got {image.shape}: {path}")
    return image


def annotation_mask(annotation: dict, shape: tuple[int, int] = (2048, 2048)) -> np.ndarray:
    """Decode MAGFiLO COCO polygons at their original resolution."""
    height, width = shape
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        rles = coco_mask.frPyObjects(segmentation, height, width)
        return coco_mask.decode(coco_mask.merge(rles)).astype(bool)
    if isinstance(segmentation["counts"], list):
        segmentation = coco_mask.frPyObjects(segmentation, height, width)
    return coco_mask.decode(segmentation).astype(bool)

