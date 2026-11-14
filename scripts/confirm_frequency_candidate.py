"""Confirm a locked recipe with date-clustered bootstrap comparisons."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_ensembles import paired


def date_paired(before, after):
    left = {r['filename']: r for r in before}
    right = {r['filename']: r for r in after}
    if len(left) != len(before) or len(right) != len(after) or set(left) != set(right):
        raise ValueError('Date bootstrap requires unique aligned observations')
    groups = defaultdict(lambda: np.zeros((2, 4), dtype=np.float64))
    fields = ('iou_sum', 'tp', 'fp', 'fn')
    for name in sorted(left):
        groups[name[:8]][0] += [left[name][k] for k in fields]
        groups[name[:8]][1] += [right[name][k] for k in fields]
    values = np.stack([groups[key] for key in sorted(groups)])
    def pq(array):
        denominator = array[..., 1] + .5 * (array[..., 2] + array[..., 3])
        return np.divide(array[..., 0], denominator, out=np.zeros_like(denominator), where=denominator > 0)
    indices = np.random.default_rng(2027).integers(len(values), size=(10000, len(values)))
    sampled = pq(values[indices].sum(axis=1))
    total = pq(values.sum(axis=0))
    return {'delta_pq': float(total[1]-total[0]), 'paired_95_interval': np.quantile(sampled[:, 1]-sampled[:, 0], [.025, .975]).tolist(),
            'unit': 'calendar date, keeping all observatories together', 'dates': len(values), 'replicates': 10000,
            'seed': 2027, 'note': 'Descriptive confirmation; adjacent dates and prior model development can remain dependent.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, default=Path('results/frequency_second_confirmation.json'))
    parser.add_argument('--lock', type=Path, default=Path('results/frequency_recipe_lock.json'))
    parser.add_argument('--output', type=Path, default=Path('results/frequency_confirmation_decision.json'))
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    report = json.loads(args.evaluation.read_text())
    protocol_path = Path('results/frequency_confirmation_protocol.json')
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != lock['confirmation_protocol_sha256']:
        raise ValueError('Confirmation protocol changed after recipe lock')
    split_path = Path('results/second_date_split.json')
    split = json.loads(split_path.read_text())
    assert report['status'] == 'complete'
    assert set(report['manifest']['filenames']) == set(split['validation_filenames'])
    assert set(report['manifest']['training_filenames']) == set(split['training_filenames'])
    assert lock['locked_unix'] < report['confirmation_started_unix']
    candidates = {c['name']: c for c in report['candidates']}
    selected = candidates[lock['selected_name']]
    control = candidates['contrast_none']
    same_tta = candidates['contrast_' + selected['tta']]
    primary = date_paired(control['observations'], selected['observations'])
    feature = date_paired(same_tta['observations'], selected['observations'])
    remaining_delta = selected['remaining_development']['pq'] - control['remaining_development']['pq']
    checks = {'positive_date_interval_vs_identity': primary['paired_95_interval'][0] > 0,
              'positive_gain_on_remaining_102': remaining_delta > 0,
              'not_worse_than_same_tta_control': feature['delta_pq'] >= -1e-12}
    summary = {'selected_name': lock['selected_name'], 'images': selected['all_development']['images'],
               'selected_pq': selected['all_development']['pq'], 'control_pq': control['all_development']['pq'],
               'remaining_delta_pq': remaining_delta, 'date_paired_vs_identity': primary,
               'date_paired_vs_same_tta_control': feature, 'checks': checks}
    payload = {'confirmed': all(checks.values()), 'summary': summary, 'evaluation': str(args.evaluation),
               'lock_sha256': hashlib.sha256(args.lock.read_bytes()).hexdigest(),
               'observation_paired_vs_identity': paired(control['observations'], selected['observations']),
               'note': 'Confirm only this preselected recipe. A failed gate does not authorize selecting a replacement from this split.'}
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
