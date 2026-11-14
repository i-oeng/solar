"""Candidates from several probability thresholds, organised as a tree, and exact selection.

``rescoring.extract_candidates`` cuts the map at one grow threshold (0.5). A filament that
the map splits at 0.5 but joins at 0.4, or merges with a neighbour at 0.5 but separates at
0.6, then has no candidate with IoU above 0.5. Cutting at several thresholds offers those
alternatives. The pieces overlap, but in a fixed way: closing and thresholding are both
increasing operations, so every component at a higher threshold lies inside exactly one
component at the next lower threshold. The candidates of one image form a forest.

Any two candidates that are not nested lie inside two different components of one
threshold, which never touch, so they stay disjoint after the one-pixel erosion. A set of
candidates with no two nested (an antichain of the forest) is therefore a valid
non-overlapping prediction, and a ground-truth filament can exceed IoU 0.5 with at most
one of them. The PQ bookkeeping of ``rescoring`` stays exact: keeping candidate c adds its
matched IoU sum to the numerator and 0.5 x views to the denominator, so with a target PQ
of 2t the best antichain maximises the sum of (mean matched IoU per view - t) over the kept
candidates. ``select_antichain`` finds that maximum exactly by dynamic programming over the
forest, using the scorer's predicted IoU in place of the unknown true one.
"""

from __future__ import annotations

import numpy as np

from .rescoring import FEATURES, extract_candidates

TREE_FEATURES = ('level', 'area_over_parent', 'children', 'largest_child_fraction')
CONTEXT_FEATURES = ('parent_p_mean', 'parent_p_q90', 'parent_log_area', 'child_max_p_mean', 'sibling_count',
                    'nodes_at_level', 'p_mean_rank_in_level', 'p_mean_minus_level_best')


def build_tree(image: np.ndarray, probability: np.ndarray, levels: tuple[float, ...], settings: dict,
               observatory: str | None = None) -> dict:
    """Candidates at each grow threshold in ``levels`` (ascending) with parent links.

    Returns per-level eroded label images, a feature matrix over all nodes (the 20 standard
    features plus TREE_FEATURES), each node's level index, its local label at that level,
    and its parent's node index (-1 for a root)."""
    levels = tuple(sorted(levels))
    labels, features, level_index, local, parent = [], [], [], [], []
    offset = 0
    previous_offset = 0
    for depth, low in enumerate(levels):
        c = extract_candidates(image, probability, low=low, high=settings['threshold'],
                               min_seed_area=settings['min_area'], max_area=settings['max_area'],
                               limb_margin=settings['limb_margin'], observatory=observatory)
        n = len(c.accepted)
        labels.append(c.labels)
        features.append(c.features)
        level_index.append(np.full(n, depth))
        local.append(np.arange(1, n + 1))
        if depth == 0:
            parent.append(np.full(n, -1))
        else:
            # Erosion is increasing too, so a child's eroded pixels lie inside its eroded parent.
            flat = c.labels.ravel()
            order = np.flatnonzero(flat)
            first = np.zeros(n + 1, np.int64)
            first[flat[order]] = order  # any pixel of a child will do
            above = labels[depth - 1].ravel()[first[1:]].astype(np.int64)
            if (above == 0).any():
                raise ValueError(f'{int((above == 0).sum())} candidates at {low} have no parent at {levels[depth - 1]}')
            parent.append(previous_offset + above - 1)
        previous_offset = offset
        offset += n
    features = np.concatenate(features) if features else np.zeros((0, len(FEATURES)))
    level_index, local, parent = np.concatenate(level_index), np.concatenate(local), np.concatenate(parent)
    area = np.expm1(features[:, FEATURES.index('log_raw_area')])
    total = len(parent)
    children = np.bincount(parent[parent >= 0], minlength=total)
    largest_child = np.zeros(total)
    np.maximum.at(largest_child, parent[parent >= 0], area[parent >= 0])
    area_over_parent = np.where(parent >= 0, area / area[np.maximum(parent, 0)], 1.0)
    extra = np.column_stack([np.asarray(levels)[level_index], area_over_parent, children, largest_child / area])
    return {'levels': levels, 'labels': labels, 'features': np.hstack([features, extra]),
            'level_index': level_index, 'local': local, 'parent': parent}


def add_context(table):
    """Add CONTEXT_FEATURES to a node table of kept nodes (columns FEATURES, TREE_FEATURES,
    ``filename``, ``depth`` and ``parent``, the row index of the parent in the same table, -1 for roots).

    They describe each candidate against its parent, its children and the other candidates of
    the same image and level, which the per-candidate features cannot see."""
    import pandas as pd
    table = table.reset_index(drop=True)
    parent = table['parent'].to_numpy()
    has = parent >= 0
    safe = np.maximum(parent, 0)
    for name, column in (('parent_p_mean', 'p_mean'), ('parent_p_q90', 'p_q90'), ('parent_log_area', 'log_raw_area'),
                         ('sibling_count', 'children')):
        values = table[column].to_numpy(np.float64)[safe]
        table[name] = np.where(has, values, np.nan)
    child_best = pd.Series(table['p_mean'].to_numpy()[has]).groupby(parent[has]).max()
    table['child_max_p_mean'] = np.nan
    table.loc[child_best.index, 'child_max_p_mean'] = child_best.to_numpy()
    group = table.groupby(['filename', 'depth'], sort=False)['p_mean']
    table['nodes_at_level'] = group.transform('size')
    table['p_mean_rank_in_level'] = group.rank(pct=True)
    table['p_mean_minus_level_best'] = table['p_mean'] - group.transform('max')
    return table


def select_antichain(gain: np.ndarray, parent: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Keep mask of the antichain with the largest total gain (nothing kept scores zero).

    ``parent`` holds node indices into the same arrays (-1 for roots) and every parent has a
    smaller ``depth`` than its children. Several images may be passed at once."""
    gain = np.asarray(gain, np.float64)
    best = np.zeros(len(gain))
    below = np.zeros(len(gain))  # best total over each node's children
    has_parent = parent >= 0
    for d in range(int(depth.max(initial=-1)), -1, -1):
        at = depth == d
        best[at] = np.maximum(gain[at], below[at])
        moving = at & has_parent
        np.add.at(below, parent[moving], best[moving])
    keep = np.zeros(len(gain), bool)
    open_ = np.zeros(len(gain), bool)  # nodes whose ancestors are all unselected
    for d in range(int(depth.max(initial=-1)) + 1):
        at = depth == d
        reachable = at & (~has_parent | open_[np.maximum(parent, 0)])
        take = reachable & (gain > 0) & (gain >= below)
        keep |= take
        open_ |= reachable & ~take
    return keep
