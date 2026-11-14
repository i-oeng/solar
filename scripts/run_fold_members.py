"""Train and evaluate the remaining annotation-round fold members.

The annotation round (scripts/run_annotation_round.py) stopped after fold 0 and half
of fold 1. This runner trains the missing members with the identical command line and
the round's frozen source snapshot (outputs/annotation_round_source), then evaluates
each on its own held-out fold with the snapshot's evaluate_checkpoint.py, exactly as
the round did. Jobs and checkpoint hashes go to results/fold_members_status.json in
the round's job format, so evaluate_fold_ensembles.py can use the new members.
Completed jobs are skipped on a rerun; a partial checkpoint is never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / 'results/fold_members_status.json'
SNAPSHOT = ROOT / 'outputs/annotation_round_source'
MEMBERS = [(2, 2026, 'most_complete'), (2, 2026, 'uniform'), (1, 2029, 'most_complete'), (1, 2029, 'uniform'),
           (2, 2029, 'most_complete'), (2, 2029, 'uniform')]


def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(state: dict) -> None:
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATUS)


def train_command(checkpoint: Path, fold: int, seed: int, selection: str) -> list[str]:
    # Identical to run_annotation_round.train_command.
    return [sys.executable, '-u', '-m', 'solar_filament.train', '--architecture', 'resnet34',
            '--encoder-weights', 'checkpoints/pretrained/resnet34-b627a593.pth', '--epochs', '30',
            '--batch-size', '2', '--workers', '2', '--seed', str(seed), '--learning-rate', '.0003',
            '--encoder-lr-factor', '.1', '--sampling', 'object', '--feature-recipe', 'contrast',
            '--split-manifest', f'results/annotation_round_folds/fold_{fold}.json',
            '--object-points', f'data/prepared/annotation_round_fold{fold}_objects',
            '--view-selection', selection, '--output', str(checkpoint)]


def run_job(state: dict, name: str, command: list[str], env: dict, checkpoint: Path | None = None) -> dict:
    done = next((j for j in state['jobs'] if j['name'] == name and j['status'] == 'complete'), None)
    if done:
        return done
    if checkpoint is not None and checkpoint.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {checkpoint}')
    state['jobs'] = [j for j in state['jobs'] if j['name'] != name]
    job = {'name': name, 'command': command, 'status': 'running', 'started_unix': time.time(), 'log': f'logs/{name}.log'}
    state['jobs'].append(job)
    save(state)
    with (ROOT / job['log']).open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        job['pid'] = process.pid
        save(state)
        code = process.wait()
    job.update(returncode=code, seconds=time.time() - job['started_unix'], status='complete' if code == 0 else 'failed')
    if code == 0 and checkpoint is not None:
        info = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if info['epoch'] != 30 or not np.isfinite(info['mean_loss']) or not all(torch.isfinite(t).all() for t in info['model'].values()):
            job.update(status='failed', error='Nonfinite or incomplete training checkpoint')
        else:
            job['checkpoint_sha256'] = digest(checkpoint)
    save(state)
    if job['status'] != 'complete':
        raise RuntimeError(f'{name} failed; see {job["log"]}')
    return job


def main() -> None:
    os.chdir(ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    source = {str(p.relative_to(SNAPSHOT)): digest(p) for p in sorted(SNAPSHOT.rglob('*.py'))}
    folds = {f'results/annotation_round_folds/fold_{k}.json': digest(f'results/annotation_round_folds/fold_{k}.json') for k in (1, 2)}
    round_status = json.loads(Path('results/annotation_round_status.json').read_text())
    for path, sha in folds.items():
        if round_status['fold_sha256'].get(path.replace('/', '\\'), round_status['fold_sha256'].get(path)) != sha:
            raise ValueError(f'Fold file differs from the annotation round: {path}')
    expected = {k.replace('\\', '/'): v for k, v in round_status['source_sha256'].items() if k.endswith('.py')}
    if any(expected.get(k) != v for k, v in source.items()):
        raise ValueError('Source snapshot differs from the annotation round record')
    if STATUS.exists():
        state = json.loads(STATUS.read_text())
        if state['source_sha256'] != source:
            raise ValueError('Snapshot changed since this runner started')
    else:
        state = {'jobs': [], 'source_sha256': source, 'fold_sha256': folds}
    state.update(status='running', pid=os.getpid(), host=platform.node(), torch=torch.__version__,
                 gpu=torch.cuda.get_device_name(), runner_sha256=digest(__file__), updated_unix=time.time())
    save(state)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SNAPSHOT / 'src'), str(ROOT / '.deps')]))
    try:
        for fold, seed, selection in MEMBERS:
            name = f'annotation_round_f{fold}_s{seed}_{selection}'
            checkpoint = Path(f'checkpoints/{name}.pt')
            run_job(state, f'{name}_train', train_command(checkpoint, fold, seed, selection), env, checkpoint)
            output = Path(f'results/{name}.json')
            run_job(state, f'{name}_evaluate', [sys.executable, '-u', str(SNAPSHOT / 'scripts/evaluate_checkpoint.py'),
                    '--checkpoint', str(checkpoint), '--settings', 'results/neural_resnet34_epoch30_eroded.json',
                    '--output', str(output), '--save-labels', f'outputs/annotation_round/{name}_masks'], env)
            if json.loads(output.read_text())['checkpoint_sha256'] != digest(checkpoint):
                raise ValueError(f'Evaluation checkpoint mismatch: {name}')
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['updated_unix'] = time.time()
        save(state)


if __name__ == '__main__':
    main()
