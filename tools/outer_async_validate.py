"""Async GEM validation: validate snapshots without blocking GPU training.

Protocol: trainer saves a snapshot (SB3 model zip) and spawns
`python -m tools.outer_async_validate run ...` via Popen (CPU-only GEM
rollouts). It polls for result.json at later ticks and adopts selections
only from steps it has actually trained past (staleness-safe: adoption
requires result.step <= trainer.num_timesteps; selection moves forward only).

Heartbeat: validator touches heartbeat.json every case; trainer treats a
stale heartbeat (>600 s) + missing result as validator failure -> falls back
to synchronous validation for that tick (never blocks selection on a dead
worker, never silently drops a validation).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.bldc.run import ROOT, sha
from benchmarks.bldc.current_rl.repeat_study import read, save, now

HEARTBEAT_TIMEOUT_S = 600.0


def snapshot_path(outdir, step):
    return Path(outdir) / '_pending' / str(step) / 'snapshot.zip'


def result_path(outdir, step):
    return Path(outdir) / '_pending' / str(step) / 'result.json'


def heartbeat_path(outdir, step):
    return Path(outdir) / '_pending' / str(step) / 'heartbeat.json'


def request_validation(python, model, outdir, step):
    """Save snapshot + spawn detached validator. Returns (Popen, Paths)."""
    pend = Path(outdir) / '_pending' / str(step)
    pend.mkdir(parents=True, exist_ok=False)
    model.save(pend / 'snapshot')
    save(pend / 'request.json', dict(step=step, requested_utc=now()))
    (pend / 'heartbeat.json').write_text(json.dumps(dict(started_utc=now())) + '\n')
    log = (pend / 'validator.log').open('x')
    proc = subprocess.Popen(
        [python, '-m', 'tools.outer_async_validate', 'run',
         '--root', str(Path(outdir).resolve()),
         '--step', str(step)],
        cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    return proc, pend


def poll(pend, proc, now_s=None):
    """Return result dict when ready, None while running, raise on failure."""
    res = pend / 'result.json'
    if res.exists():
        return json.loads(res.read_text())
    if proc.poll() is not None:
        raise RuntimeError(f'Validator exited rc={proc.returncode} without result (see validator.log)')
    hb = pend / 'heartbeat.json'
    try:
        age = (now_s or time.time()) - hb.stat().st_mtime
    except FileNotFoundError:
        age = 0.0
    if age > HEARTBEAT_TIMEOUT_S:
        raise TimeoutError(f'Validator heartbeat stale ({age:.0f}s)')
    return None


def adoptable(result, trainer_steps, last_adopted):
    """Staleness-safe adoption predicate (pure, unit-tested)."""
    if result is None:
        return False
    step = result.get('step', -1)
    return step > last_adopted and step <= trainer_steps


def run_validator(root, step):
    """Entry point for the detached process. Imports SB3/GEM here (CPU)."""
    from stable_baselines3 import DDPG
    from tools.outer_rl import evaluation
    from tools import outer_warp_ceiling as ceiling
    root = Path(root)
    p = ceiling.verify(root, _discover_protocol(root))
    cfg = read(root / 'config.json')
    inner = DDPG.load(root / 'inner_model.zip', device='cpu')
    inner.policy.set_training_mode(False)
    pend = root / '_pending' / str(step)
    (pend / 'heartbeat.json').write_text(json.dumps(dict(state='started', utc=now())) + '\n')
    model = DDPG.load(pend / 'snapshot.zip', device='cpu')
    rows = evaluation.compare(model, cfg['plant'], cfg['inner_study'], inner,
                              cfg['validation_cases'], pend / 'validation',
                              baseline_directory=root / 'baselines')
    value = ceiling.score(rows)
    if not np.isfinite(value[-1]):
        raise RuntimeError('Nonfinite validation RMSE')
    (pend / 'heartbeat.json').write_text(json.dumps(dict(state='done', utc=now())) + '\n')
    save(pend / 'result.json', dict(step=step, score=value,
                                    passed_cases=sum(r['passed'] for r in rows),
                                    case_count=len(rows), completed_utc=now()))
    return value


def promote(pend, dest):
    """Atomically move validated artifacts into validation/<step>/."""
    import shutil
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    shutil.move(str(Path(pend) / 'validation'), str(dest))
    return dest


def _discover_protocol(root):
    proto = read(Path(root) / 'protocol.json')
    return proto.get('protocol', ceiling.PROTOCOL)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['run'])
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--step', type=int, required=True)
    a = ap.parse_args()
    run_validator(a.root, a.step)


if __name__ == '__main__':
    main()
