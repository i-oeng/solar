"""Run the frozen annotation-view screen and conditional paired-seed replication.

No Kaggle operations. Source is snapshotted before training. Status and logs survive
the interactive session; incomplete checkpoints are never mistaken for completed runs.
"""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from confirm_frequency_candidate import date_paired
from evaluate_ensembles import summarize


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / 'results/annotation_round_status.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(state):
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATUS)
    lines = ['# Annotation-view experiment status', '', f'Current state: **{state["status"]}**.', '',
        'Protocol: [next_round_2026_11_05.md](next_round_2026_11_05.md). '
        'Inference findings: [instance_round_results_2026_11_05.md](instance_round_results_2026_11_05.md).', '',
        'Training uses a frozen source snapshot and the same-host matched control. '
        'This controller performs no Kaggle upload or accepted-artifact replacement.', '']
    if 'screen' in state:
        screen = state['screen']
        full, remaining = screen['all_development'], screen['remaining_development']
        lo, hi = full['paired_95_interval']
        lines += ['## Screening result', '',
            f'Full-development PQ change: **{full["delta_pq"]:+.7f}**, date-paired 95% interval [{lo:+.7f}, {hi:+.7f}].', '',
            f'Change on the 102 non-calibration images: **{remaining["delta_pq"]:+.7f}**. '
            f'Replication gate: **{"passed" if screen["passed"] else "failed"}**.', '',
            'A failed screen stops this training recipe. A passing screen starts the prescribed three-fold, two-seed comparison; it does not establish a confirmed improvement.', '']
    if 'replication' in state:
        result = state['replication']
        lo, hi = result['paired_95_interval']
        lines += ['## Replication result', '',
            f'Mean pooled out-of-fold gain across seeds: **{result["mean_delta_pq"]:+.7f}**, '
            f'date-paired 95% interval [{lo:+.7f}, {hi:+.7f}]. Gate: **{"passed" if result["passed"] else "failed"}**.', '',
            'These are internal development replication results, not an independent test score.', '']
    if 'error' in state:
        lines += ['## Failure', '', state['error'], '']
    lines += ['## Recent jobs', '']
    for job in state.get('jobs', [])[-4:]:
        lines.append(f'- `{job["name"]}`: {job["status"]}; log `{job["log"]}`.')
    lines += ['', 'Machine-readable provenance and complete job history: `results/annotation_round_status.json`.']
    (ROOT / 'docs/annotation_round_status.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def pooled(rows):
    return summarize([dict(row, by_size={}) for row in rows])


def seed_date_comparison(pairs):
    """Bootstrap dates jointly for both seeds; seed runs are not independent images."""
    fields = ('iou_sum', 'tp', 'fp', 'fn')
    seeds = sorted(pairs)
    grouped = defaultdict(lambda: np.zeros((len(seeds), 2, 4), np.float64))
    expected = None
    per_seed = {}
    for index, seed in enumerate(seeds):
        before, after = pairs[seed]
        left, right = ({r['filename']: r for r in rows} for rows in (before, after))
        if len(left) != len(before) or len(right) != len(after) or set(left) != set(right):
            raise ValueError('Each seed needs unique aligned out-of-fold observations')
        if expected is not None and set(left) != expected:
            raise ValueError('Every seed must evaluate the same observations')
        expected = set(left)
        for name in sorted(left):
            grouped[name[:8]][index, 0] += [left[name][field] for field in fields]
            grouped[name[:8]][index, 1] += [right[name][field] for field in fields]
        per_seed[str(seed)] = date_paired(before, after)
    values = np.stack([grouped[d] for d in sorted(grouped)])
    def pq(array):
        denominator = array[..., 1] + .5 * (array[..., 2] + array[..., 3])
        return np.divide(array[..., 0], denominator, out=np.zeros_like(denominator), where=denominator > 0)
    rng = np.random.default_rng(2027)
    differences = []
    # Chunk resampling to bound memory on the laptop.
    for _ in range(100):
        indices = rng.integers(len(values), size=(100, len(values)))
        scores = pq(values[indices].sum(axis=1))
        differences.extend((scores[..., 1] - scores[..., 0]).mean(axis=1).tolist())
    scores = pq(values.sum(axis=0))
    mean_gain = float((scores[:, 1] - scores[:, 0]).mean())
    interval = np.quantile(differences, [.025, .975]).tolist()
    return {'mean_delta_pq': mean_gain, 'paired_95_interval': interval, 'per_seed': per_seed,
        'dates': len(values), 'images_per_seed': len(expected), 'replicates': 10000, 'seed': 2027,
        'passed': all(r['delta_pq'] > 0 for r in per_seed.values()) and mean_gain >= .005 and interval[0] > 0,
        'note': 'Date resampling retains all observatories and paired seeds. Internal replication, not independent testing.'}


def check_screen(control_rows, candidate_rows, calibration):
    full = date_paired(control_rows, candidate_rows)
    remaining_control = [r for r in control_rows if r['filename'] not in calibration]
    remaining_candidate = [r for r in candidate_rows if r['filename'] not in calibration]
    remaining = date_paired(remaining_control, remaining_candidate)
    return {'passed': full['delta_pq'] > 0 and remaining['delta_pq'] > 0,
            'all_development': full, 'remaining_development': remaining,
            'note': 'Screening advances replication only. No promotion or submission.'}


def run_job(state, name, command, env, checkpoint=None):
    if checkpoint is not None and checkpoint.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {checkpoint}')
    job = {'name': name, 'command': command, 'status': 'running', 'started_unix': time.time(),
           'log': f'logs/{name}.log'}
    state['jobs'].append(job)
    save(state)
    with (ROOT / job['log']).open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        job['pid'] = process.pid
        save(state)
        code = process.wait()
    job.update(returncode=code, seconds=time.time()-job['started_unix'], status='complete' if code == 0 else 'failed')
    if code == 0 and checkpoint is not None:
        info = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if info['epoch'] != 30 or not np.isfinite(info['mean_loss']) or not all(torch.isfinite(t).all() for t in info['model'].values()):
            job.update(status='failed', error='Nonfinite or incomplete training checkpoint')
            save(state)
            raise ValueError(job['error'])
        job['checkpoint_sha256'] = digest(checkpoint)
    save(state)
    if code:
        raise RuntimeError(f'{name} failed with exit code {code}; see {job["log"]}')


def train_command(checkpoint, split, seed, selection, object_cache):
    return [sys.executable, '-u', '-m', 'solar_filament.train', '--architecture', 'resnet34',
        '--encoder-weights', 'checkpoints/pretrained/resnet34-b627a593.pth', '--epochs', '30',
        '--batch-size', '2', '--workers', '2', '--seed', str(seed), '--learning-rate', '.0003',
        '--encoder-lr-factor', '.1', '--sampling', 'object', '--feature-recipe', 'contrast',
        '--split-manifest', str(split), '--object-points', str(object_cache),
        '--view-selection', selection, '--output', str(checkpoint)]


def evaluate(state, snapshot, env, name, checkpoint):
    output = Path(f'results/{name}.json')
    labels = Path(f'outputs/annotation_round/{name}_masks')
    run_job(state, f'{name}_evaluate', [sys.executable, '-u', str(snapshot / 'scripts/evaluate_checkpoint.py'),
        '--checkpoint', str(checkpoint), '--settings', 'results/neural_resnet34_epoch30_eroded.json',
        '--output', str(output), '--save-labels', str(labels)], env)
    report = json.loads(output.read_text())
    if report['checkpoint_sha256'] != digest(checkpoint):
        raise ValueError('Evaluation checkpoint mismatch')
    return next(c for c in report['candidates'] if c['erosion_radius'] == 1)


def verify_control():
    reference = json.loads(Path('results/frequency_second_confirmation.json').read_text())
    path = Path('checkpoints/second_contrast.pt')
    if digest(path) != reference['manifest']['models']['contrast']['sha256']:
        raise ValueError('Screening control changed')
    info = torch.load(path, map_location='cpu', weights_only=True)
    expected = dict(epoch=30, seed=2026, samples_per_epoch=1130, learning_rate=.0003,
        encoder_lr_factor=.1, architecture='resnet34', feature_recipe='contrast', sampling='object')
    if any(info.get(k) != v for k, v in expected.items()):
        raise ValueError('Control training recipe does not match the protocol')
    previous = json.loads(Path('results/second_split_training_status.json').read_text())
    run = next(r for r in previous['runs'] if r['recipe'] == 'contrast')
    if previous['host'] != platform.node() or run['status'] != 'complete' or run['checkpoint_sha256'] != digest(path):
        raise ValueError('Control host/provenance mismatch')
    for flag, value in [('--batch-size', '2'), ('--epochs', '30'), ('--seed', '2026'), ('--workers', '2')]:
        if run['command'][run['command'].index(flag)+1] != value:
            raise ValueError('Control command mismatch')
    for module in ('model.py', 'features.py', 'data.py', 'sampling.py'):
        filename = Path('src/solar_filament') / module
        if digest(filename) != previous['source_sha256'][str(filename)]:
            raise ValueError(f'Control-relevant source changed: {module}')
    return reference


def main():
    os.chdir(ROOT)
    if STATUS.exists():
        raise FileExistsError('This experiment already has a status record; inspect it before restarting')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the scheduled training budget')
    reference = verify_control()
    state = {'status': 'preparing', 'pid': os.getpid(), 'started_unix': time.time(),
        'protocol_sha256': digest('docs/next_round_2026_11_05.md'), 'jobs': [],
        'host': platform.node(), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'control_checkpoint_sha256': digest('checkpoints/second_contrast.pt')}
    # Exclusive creation prevents two supervisors starting the same experiment.
    with STATUS.open('x') as handle:
        json.dump(state, handle, indent=2)
    try:
        snapshot = ROOT / 'outputs/annotation_round_source'
        if snapshot.exists():
            raise FileExistsError(snapshot)
        for directory in ('src/solar_filament', 'scripts'):
            target = snapshot / directory
            target.mkdir(parents=True, exist_ok=True)
            for path in (ROOT / directory).glob('*.py'):
                shutil.copy2(path, target / path.name)
        shutil.copy2('docs/next_round_2026_11_05.md', snapshot / 'protocol.md')
        state['source_sha256'] = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*') if p.is_file()}
        state['fold_sha256'] = {str(p): digest(p) for p in sorted(Path('results/annotation_round_folds').glob('fold_*.json'))}
        if len(state['fold_sha256']) != 3:
            raise ValueError('Three frozen folds required before training')
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(snapshot / 'src'), str(ROOT / '.deps')]))
        state['status'] = 'waiting_for_audit_and_cache'
        save(state)
        deadline = time.monotonic() + 3600
        while not Path('results/instance_round_audit.json').exists() or not Path('data/prepared/annotation_views_v1/manifest.json').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Audit/cache did not finish within one hour; training did not start')
            time.sleep(5)
        audit = json.loads(Path('results/instance_round_audit.json').read_text())
        if audit['status'] != 'complete' or not audit['baseline_exactly_reproduced'] or audit['manifest']['protocol_sha256'] != state['protocol_sha256']:
            raise ValueError('Audit prerequisite failed')
        state['audit_sha256'] = digest('results/instance_round_audit.json')
        state['status'] = 'screening_training'
        save(state)
        checkpoint = Path('checkpoints/annotation_round_second_uniform.pt')
        run_job(state, 'annotation_round_second_uniform_train', train_command(checkpoint,
            Path('results/second_date_split.json'), 2026, 'uniform', Path('data/prepared/object_points_second')), env, checkpoint)
        candidate = evaluate(state, snapshot, env, 'annotation_round_second_uniform', checkpoint)
        control = next(c for c in reference['candidates'] if c['name'] == 'contrast_none')
        screen = check_screen(control['observations'], candidate['observations'], set(reference['manifest']['calibration_filenames']))
        state['screen'] = screen
        Path('results/annotation_round_screen.json').write_text(json.dumps({'control': control['all_development'],
            'candidate': {k: v for k, v in candidate.items() if k != 'observations'}, 'screen': screen}, indent=2) + '\n')
        save(state)
        if not screen['passed']:
            state.update(status='complete_screen_failed', completed_unix=time.time())
            save(state)
            return
        state['status'] = 'replicating'
        save(state)
        pairs = {seed: ([], []) for seed in (2026, 2029)}
        fold_reports = []
        for fold in range(3):
            split = Path(f'results/annotation_round_folds/fold_{fold}.json')
            if digest(split) != state['fold_sha256'][str(split)]:
                raise ValueError('Frozen folds changed')
            object_cache = Path(f'data/prepared/annotation_round_fold{fold}_objects')
            run_job(state, f'annotation_round_fold{fold}_cache', [sys.executable, '-u', '-m', 'solar_filament.sampling',
                '--output', str(object_cache), '--split-manifest', str(split)], env)
            for seed in (2026, 2029):
                fold_pair = []
                for index, selection in enumerate(('most_complete', 'uniform')):
                    name = f'annotation_round_f{fold}_s{seed}_{selection}'
                    checkpoint = Path(f'checkpoints/{name}.pt')
                    run_job(state, name + '_train', train_command(checkpoint, split, seed, selection, object_cache), env, checkpoint)
                    report = evaluate(state, snapshot, env, name, checkpoint)
                    pairs[seed][index].extend(report['observations'])
                    fold_pair.append(report)
                fold_reports.append({'fold': fold, 'seed': seed, 'comparison': date_paired(fold_pair[0]['observations'], fold_pair[1]['observations']),
                    'control': {k: v for k, v in fold_pair[0].items() if k != 'observations'},
                    'candidate': {k: v for k, v in fold_pair[1].items() if k != 'observations'}})
        comparison = seed_date_comparison(pairs)
        Path('results/annotation_round_replication.json').write_text(json.dumps({'comparison': comparison,
            'folds': fold_reports, 'protocol_sha256': state['protocol_sha256']}, indent=2) + '\n')
        state.update(status='complete_replication_passed' if comparison['passed'] else 'complete_replication_failed',
                     replication=comparison, completed_unix=time.time())
        save(state)
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', failed_unix=time.time())
        save(state)
        raise


if __name__ == '__main__':
    main()
