"""Observation-level Panoptic Quality on separate MAGFiLO annotator views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import annotation_mask


@dataclass
class PQCounts:
    iou_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def pq(self) -> float:
        denominator = self.tp + 0.5 * (self.fp + self.fn)
        return self.iou_sum / denominator if denominator else 0.0

    def __iadd__(self, other: "PQCounts") -> "PQCounts":
        self.iou_sum += other.iou_sum
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self


def score_view(labels: np.ndarray, annotations: list[dict], threshold: float = 0.5) -> PQCounts:
    """Count all IoU > 0.5 hits and unmatched objects.

    The input labels are zero for background and positive for instances. Matching
    is performed per annotator view, as in the organizers' self-evaluation notebook.
    """
    areas = np.bincount(labels.ravel())
    prediction_ids = np.flatnonzero(areas[1:]) + 1
    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    iou_sum = 0.0
    tp = 0
    for gt_index, annotation in enumerate(annotations):
        gt = annotation_mask(annotation, labels.shape)
        gt_area = int(gt.sum())
        intersections = np.bincount(labels[gt], minlength=len(areas))
        for prediction_id in np.flatnonzero(intersections[1:]) + 1:
            intersection = int(intersections[prediction_id])
            union = gt_area + int(areas[prediction_id]) - intersection
            iou = intersection / union if union else 0.0
            if iou > threshold:
                matched_gt.add(gt_index)
                matched_predictions.add(int(prediction_id))
                iou_sum += iou
                tp += 1
    return PQCounts(
        iou_sum=iou_sum,
        tp=tp,
        fp=len(prediction_ids) - len(matched_predictions),
        fn=len(annotations) - len(matched_gt),
    )


def score_observation(labels: np.ndarray, views: list[list[dict]]) -> PQCounts:
    total = PQCounts()
    for annotations in views:
        total += score_view(labels, annotations)
    return total
