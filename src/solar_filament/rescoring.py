"""Filament candidates from a probability map, their features, and exact PQ bookkeeping.

The accepted postprocessing keeps a component of the closed 0.5-probability mask
only if it contains a 0.98-probability seed of 400 to 40000 pixels and is itself at
most 40000 pixels; one-pixel erosion follows. That rule is a hand-written classifier
over components. Here every component becomes a candidate with features, so a
learned scorer can make the same keep/drop decision. Candidate masks are exactly the
masks the accepted rule would output for that component.

Candidates are distinct 8-connected components, so they never touch and one-pixel
erosion acts on each independently. A ground-truth filament can then exceed IoU 0.5
with at most one candidate, which makes the PQ of any kept subset an exact sum of
per-candidate terms. Adding a candidate seen in n annotation views always grows the
PQ denominator by 0.5 * n (each view gains either a TP that removes an FN, or an FP),
and grows the numerator by its summed matched IoU. Keeping it raises PQ exactly when
its mean matched IoU per view exceeds PQ / 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .baseline import solar_disk_mask
from .refine import erode_labels

OBSERVATORIES = ('Bh', 'Ch', 'Lh', 'Mh', 'Th', 'Uh')
FEATURES = ('log_area', 'log_raw_area', 'p_max', 'p_mean', 'p_q50', 'p_q90', 'frac_ge_090', 'frac_ge_098',
            'log_seed_pixels', 'log_largest_seed', 'accepted_rule', 'fill', 'elongation', 'log_length',
            'radial', 'contrast', 'ring_p_mean', 'log_candidates_in_image', 'accepted_in_image', 'observatory')


@dataclass
class Candidates:
    labels: np.ndarray            # uint16, eroded candidate masks, 0 = background
    features: np.ndarray          # (n, len(FEATURES)) float64
    accepted: np.ndarray          # (n,) bool, the hand-written rule


def _disk_geometry(image: np.ndarray) -> tuple[float, float, float]:
    full = solar_disk_mask(image, 0)
    ys, xs = np.nonzero(full)
    return float(ys.mean()), float(xs.mean()), float(np.sqrt(len(ys) / np.pi))


def extract_candidates(image: np.ndarray, probability: np.ndarray, *, low: float = 0.5, high: float = 0.98,
                       min_seed_area: int = 400, max_area: int = 40000, limb_margin: int = 65,
                       min_area: int = 30, erosion: int = 1, observatory: str | None = None) -> Candidates:
    disk = solar_disk_mask(image, limb_margin)
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    strong = ((probability >= high) & disk).astype(np.uint8)
    weak = cv2.morphologyEx(((probability >= low) & disk).astype(np.uint8), cv2.MORPH_CLOSE, ellipse)
    count, components, stats, _ = cv2.connectedComponentsWithStats(weak, 8)
    s_count, s_components, s_stats, _ = cv2.connectedComponentsWithStats(strong, 8)
    # Each strong component lies inside exactly one weak component (closing only adds pixels).
    s_parent = np.zeros(s_count, np.int64)
    if s_count > 1:
        first = np.zeros(s_count, np.int64)
        flat_s = s_components.ravel()
        order = np.flatnonzero(flat_s)
        first[flat_s[order]] = order
        s_parent[1:] = components.ravel()[first[1:]]
    s_area = s_stats[:, cv2.CC_STAT_AREA]
    largest_seed = np.zeros(count, np.int64)
    valid_seed = np.zeros(count, bool)
    for index in range(1, s_count):
        parent = s_parent[index]
        largest_seed[parent] = max(largest_seed[parent], s_area[index])
        valid_seed[parent] |= min_seed_area <= s_area[index] <= max_area
    area = stats[:, cv2.CC_STAT_AREA]
    accepted_rule = valid_seed & (area <= max_area)
    keep = np.flatnonzero(((area >= min_area) | accepted_rule) & (np.arange(count) > 0))
    lookup = np.zeros(count, np.int32)
    lookup[keep] = np.arange(1, len(keep) + 1)
    raw = lookup[components]
    if len(keep) > 65535:
        raise ValueError('Too many candidates for uint16 labels')
    labels = erode_labels(raw.astype(np.uint16), erosion)
    eroded_area = np.bincount(labels.ravel(), minlength=len(keep) + 1)
    survivors = np.flatnonzero(eroded_area[1:] > 0) + 1
    # Renumber survivors contiguously; a candidate erased by erosion cannot be output.
    remap = np.zeros(len(keep) + 1, np.uint16)
    remap[survivors] = np.arange(1, len(survivors) + 1)
    labels = remap[labels]
    keep = keep[survivors - 1]

    cy, cx, radius = _disk_geometry(image)
    median = float(np.median(image[disk])) if disk.any() else 1.0
    seed_pixels = np.bincount(components.ravel()[strong.ravel() > 0], minlength=count)
    accepted_count = int(accepted_rule[keep].sum())
    rows = []
    for new_id, old_id in enumerate(keep, 1):
        x, y, w, h = (int(v) for v in stats[old_id, :4])
        pad = 15
        y0, y1, x0, x1 = max(y - pad, 0), min(y + h + pad, image.shape[0]), max(x - pad, 0), min(x + w + pad, image.shape[1])
        component = components[y0:y1, x0:x1] == old_id
        values = probability[y0:y1, x0:x1][component]
        ring = cv2.dilate(component.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1,) * 2)) > 0
        ring &= ~component & disk[y0:y1, x0:x1]
        inside_intensity = float(image[y0:y1, x0:x1][component].mean())
        ring_intensity = float(image[y0:y1, x0:x1][ring].mean()) if ring.any() else inside_intensity
        points = np.column_stack(np.nonzero(component)[::-1]).astype(np.float32)
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(points)
        length, width = max(side_a, side_b, 1.0), max(min(side_a, side_b), 1.0)
        ys, xs = np.nonzero(component)
        distance = np.hypot(ys.mean() + y0 - cy, xs.mean() + x0 - cx) / radius
        site = observatory if observatory in OBSERVATORIES else None
        rows.append([
            np.log1p(eroded_area[survivors[new_id - 1]]), np.log1p(area[old_id]),
            float(values.max()), float(values.mean()), float(np.quantile(values, .5)), float(np.quantile(values, .9)),
            float((values >= .9).mean()), float((values >= high).mean()),
            np.log1p(seed_pixels[old_id]), np.log1p(largest_seed[old_id]), float(accepted_rule[old_id]),
            area[old_id] / max(w * h, 1), length / width, np.log1p(length), distance,
            (ring_intensity - inside_intensity) / max(median, 1.0),
            float(probability[y0:y1, x0:x1][ring].mean()) if ring.any() else 0.0,
            np.log1p(len(keep)), float(accepted_count), float(OBSERVATORIES.index(site)) if site else -1.0,
        ])
    features = np.asarray(rows, np.float64).reshape(len(keep), len(FEATURES))
    return Candidates(labels=labels, features=features, accepted=accepted_rule[keep].astype(bool))


def match_candidates(labels: np.ndarray, views: list[list[np.ndarray]]) -> dict[str, np.ndarray | int]:
    """Per-candidate matched IoU sums, hit counts and matched-view counts over annotation views.

    ``views`` holds, per annotator view, the flat pixel indices of each annotated filament.
    """
    flat = labels.ravel()
    n = int(labels.max())
    areas = np.bincount(flat, minlength=n + 1)
    iou_sum = np.zeros(n + 1)
    hits = np.zeros(n + 1, np.int64)
    matched_views = np.zeros(n + 1, np.int64)
    for view in views:
        matched_here = np.zeros(n + 1, bool)
        for pixels in view:
            overlap = np.bincount(flat[pixels], minlength=n + 1)
            iou = overlap / np.maximum(len(pixels) + areas - overlap, 1)
            found = np.flatnonzero(iou[1:] > 0.5) + 1
            iou_sum[found] += iou[found]
            hits[found] += 1
            matched_here[found] = True
        matched_views += matched_here
    return {'iou_sum': iou_sum[1:], 'hits': hits[1:], 'matched_views': matched_views[1:],
            'views': len(views), 'ground_truth': sum(len(view) for view in views)}


def subset_counts(iou_sum, hits, matched_views, views, keep, ground_truth) -> dict:
    """Exact pooled PQ counts for keeping the candidates where ``keep`` is True."""
    keep = np.asarray(keep, bool)
    tp = int(hits[keep].sum())
    return {'iou_sum': float(iou_sum[keep].sum()), 'tp': tp,
            'fp': int((views[keep] - matched_views[keep]).sum()), 'fn': int(ground_truth - tp)}
