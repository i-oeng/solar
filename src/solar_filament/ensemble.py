"""Sequential square-symmetry inference; combine probabilities, never labels."""

from __future__ import annotations

import numpy as np


TRANSFORMS = ('identity', 'horizontal', 'vertical', 'both',
              'transpose', 'transpose_horizontal', 'transpose_vertical', 'transpose_both')
TTA_COUNTS = {'none': 1, 'flip4': 4, 'd4': 8}


def transform_image(image: np.ndarray, index: int, inverse: bool = False) -> np.ndarray:
    """Transpose then flip; the inverse applies the operations in reverse order."""
    if index not in range(8):
        raise ValueError('Symmetry index must be in [0, 7]')
    result = image
    if index >= 4 and not inverse:
        result = result.T
    if index % 4 in (1, 3):
        result = np.fliplr(result)
    if index % 4 in (2, 3):
        result = np.flipud(result)
    if index >= 4 and inverse:
        result = result.T
    return np.ascontiguousarray(result)


def average_probabilities(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    if first.shape != second.shape or not 0 <= weight <= 1:
        raise ValueError('Aligned shapes and a weight in [0, 1] are required')
    # Preserve controls bit for bit, including dtype.
    if weight == 1:
        return first.copy()
    if weight == 0:
        return second.copy()
    return first * np.float32(weight) + second * np.float32(1 - weight)
