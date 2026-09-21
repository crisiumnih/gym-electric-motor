"""Outer TD3 learner swap (v1 contract/obs/reward unchanged), Warp training.

Literature + trace evidence both indict vanilla DDPG (chatter/stall,
overestimation): TD3's twin critics + delayed actors + target smoothing are
the cited fix (overshoot -70%, SSE <0.5%). ONE variable: the learner.
Gamma .995, L1, -5200, 2x128/2x256, v1 memory, frozen inner, fresh seeds 26/27.
TD3-specific hypers are SB3 defaults (policy_delay=2, target_noise 0.2/0.5),
pre-declared here. Warp trains (8 envs, 40-case cycling), GEM validates.

Lineage: outer-td3-v1. Frozen GEM sources imported, never edited.
"""
import argparse
from pathlib import Path

import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from benchmarks.bldc.run import sha, ROOT
from benchmarks.bldc.current_rl.repeat_study import read, save
from tools.outer_rl import study as original
from tools import outer_warp_ceiling as ceiling

PROTOCOL = 'outer-td3-v1'
LABEL = '128x2'
SEEDS = (26, 27)
BUDGET = 250000
VALID_EVERY = 25000
N_ENVS = 8
# TD3 algorithm hypers (SB3 defaults, pre-declared as part of the learner):
POLICY_DELAY = 2
TARGET_POLICY_NOISE = 0.2
TARGET_NOISE_CLIP = 0.5


def td3_learner(env, widths, seed, cfg):
    """Mirror of ceiling.learner with TD3; all shared hypers from config."""
    c = cfg['learner']
    import numpy as np
    return TD3('MlpPolicy', env, seed=seed, device=c.get('device', 'cuda'), learning_rate=c['learning_rate'],
        gamma=c['gamma'], tau=c['tau'], buffer_size=c['buffer_size'], learning_starts=c['learning_starts'],
        batch_size=c['batch_size'], train_freq=c['train_freq'], gradient_steps=c['gradient_steps'],
        action_noise=NormalActionNoise(np.zeros(1), np.ones(1) * c['noise_sigma']),
        policy_kwargs=dict(net_arch=dict(pi=widths, qf=c['critic_widths']), activation_fn=torch.nn.ReLU),
        policy_delay=POLICY_DELAY, target_policy_noise=TARGET_POLICY_NOISE,
        target_noise_clip=TARGET_NOISE_CLIP)


def prepare(output, inner, budget=BUDGET, seeds=SEEDS, validation_every=VALID_EVERY,
            max_workers=2, learner_extra=None):
    output = Path(output).resolve()
    p = ceiling.prepare(output, inner, budget=budget, seeds=list(seeds), widths=(LABEL,),
                        validation_every=validation_every, max_workers=max_workers,
                        protocol_name=PROTOCOL, n_envs=N_ENVS, learner_extra=learner_extra,
                        interpretation_note='; td3 learner swap, v1 contract/obs/reward')
    cfg = read(output / 'config.json')
    cfg['learner']['td3'] = dict(policy_delay=POLICY_DELAY, target_policy_noise=TARGET_POLICY_NOISE,
                                 target_noise_clip=TARGET_NOISE_CLIP)
    save(output / 'config.json', cfg)
    for key in ('protocol.json',):
        proto = read(output / key)
        proto['config'] = cfg
        import hashlib
        import json
        proto['config_sha256'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        proto['frozen_files']['config.json'] = sha(output / 'config.json')
        proto['source_sha256'][str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
        save(output / key, proto)
    (output / 'protocol.sha256').write_text(sha(output / 'protocol.json') + '\n')
    return read(output / 'protocol.json')


def verify(output):
    p = ceiling.verify(output, PROTOCOL)
    if sha(Path(__file__)) != p['source_sha256'].get(str(Path(__file__).relative_to(ROOT))):
        raise ValueError('Changed td3 source')
    c = read(Path(output) / 'config.json')
    td3 = c['learner'].get('td3', {})
    if (td3.get('policy_delay'), td3.get('target_policy_noise'), td3.get('target_noise_clip')) != (
            POLICY_DELAY, TARGET_POLICY_NOISE, TARGET_NOISE_CLIP):
        raise ValueError('Wrong TD3 dispatch')
    if c['contract'].get('version') != 'bldc-outer-speed-v1':
        raise ValueError('TD3 screen must keep v1 contract')
    return p


def run(output, label, seed):
    from unittest.mock import patch as _patch
    with _patch.object(ceiling, 'learner', td3_learner):
        ceiling.run_arm(Path(output), label, seed, PROTOCOL)


def execute(output, workers=None):
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from benchmarks.bldc.current_rl.repeat_study import now
    output = Path(output).resolve()
    p = verify(output)
    workers = workers or p.get('max_workers', 2)
    with (output / 'queue_started.json').open('x') as f:
        import json
        json.dump(dict(started_utc=now()), f)
    (output / 'logs').mkdir(exist_ok=False)
    original.atomic(output / 'status.json', dict(status='preparing_baselines'))
    try:
        original.torch.set_num_threads(1)
        original.torch.use_deterministic_algorithms(True)
        cfg = read(output / 'config.json')
        from stable_baselines3 import DDPG
        inner = DDPG.load(output / 'inner_model.zip', device='cpu')
        from tools import outer_horizon_study as horizon
        with horizon.environment('g0995'):
            from tools.outer_rl import evaluation
            evaluation.baselines(cfg['plant'], cfg['inner_study'], inner, cfg['validation_cases'], output / 'baselines')
        original.atomic(output / 'status.json', dict(status='running'))

        def job(r):
            started = now()
            try:
                with (output / 'logs' / f"{r['arm']}.log").open('x') as log:
                    process = subprocess.run([sys.executable, '-m', 'tools.outer_td3_v1', 'run', '--output', str(output),
                        '--label', r['label'], '--seed', str(r['seed'])], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                return dict(**r, returncode=process.returncode, started_utc=started, completed_utc=now())
            except Exception as exc:
                return dict(**r, returncode=-1, failure=f'{type(exc).__name__}: {exc}')
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(job, r) for r in p['records']]
            for future in as_completed(futures):
                results.append(future.result())
                original.atomic(output / 'queue_status.json', dict(completed_jobs=len(results), total_jobs=len(p['records']), results=results))
                print(__import__('json').dumps(results[-1]), flush=True)
        save(output / 'run_summary.json', results)
        if any(r['returncode'] for r in results):
            raise RuntimeError('Failed job; no aggregate selection')
        select(output)
        original.atomic(output / 'status.json', dict(status='completed', completed_utc=now()))
    except BaseException as exc:
        original.atomic(output / 'status.json', dict(status='failed', failure=f'{type(exc).__name__}: {exc}', failed_utc=now()))
        raise


def select(output):
    return ceiling.select(Path(output), PROTOCOL)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['prepare', 'run', 'execute', 'select'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inner', type=Path, default=Path('results/bldc/outer-inner-freeze-v1'))
    ap.add_argument('--label', default=LABEL)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare(a.output, a.inner)
    elif a.command == 'run':
        run(a.output, a.label, a.seed)
    elif a.command == 'execute':
        execute(a.output, a.workers)
    else:
        select(a.output)


if __name__ == '__main__':
    main()
