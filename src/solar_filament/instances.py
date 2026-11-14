"""Experimental seed-preserving instance separation; legacy inference stays fixed."""

import cv2
import numpy as np
from skimage.segmentation import watershed

from .baseline import components_to_labels, solar_disk_mask
from .inference import hysteresis_to_labels


def probability_stages(image, probability, settings):
    disk = solar_disk_mask(image, settings['limb_margin'])
    strong = ((probability >= settings['threshold']) & disk).astype(np.uint8)
    seeds = components_to_labels(strong, settings['min_area'], settings['max_area'])
    weak = ((probability >= settings['grow_threshold']) & disk).astype(np.uint8)
    weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, regions, stats, _ = cv2.connectedComponentsWithStats(weak, 8)
    seeded = np.bincount(regions[seeds > 0], minlength=len(stats)) > 0
    valid = seeded & (stats[:, cv2.CC_STAT_AREA] <= settings['max_area'])
    valid[0] = False
    lookup = np.zeros(len(stats), np.int32)
    lookup[valid] = np.arange(1, np.count_nonzero(valid) + 1)
    labels = lookup[regions]
    return {'disk': disk, 'seeds': seeds, 'regions': regions,
            'region_areas': stats[:, cv2.CC_STAT_AREA], 'seeded': seeded,
            'labels': labels}


def separate_seeds(probability, seeds, labels):
    """Partition accepted foreground without adding support or accepting new seeds."""
    if probability.shape != labels.shape or seeds.shape != labels.shape:
        raise ValueError('Probability, seeds and accepted labels must align')
    support = labels > 0
    if not support.any():
        return np.zeros_like(labels, dtype=np.int32)
    markers = np.where(support, seeds, 0).astype(np.int32)
    result = watershed(-probability, markers, connectivity=np.ones((3, 3), bool),
                       mask=support).astype(np.int32)
    if not np.array_equal(result > 0, support):
        raise ValueError('Watershed changed accepted foreground support')
    return result


def verify_legacy_stages(image, probability, settings, stages):
    expected = hysteresis_to_labels(image, probability, settings['threshold'],
        settings['grow_threshold'], settings['min_area'], settings['max_area'],
        settings['limb_margin'])
    if not np.array_equal(stages['labels'], expected):
        raise AssertionError('Instrumented stages differ from legacy inference')
