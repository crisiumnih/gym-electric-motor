"""Paired discount-horizon rerun with CUDA outer updates (v2).

Same matched design as `tools.outer_horizon_study` (L1 tracking, -5200 failure
penalty, gamma .995 vs .999, fresh seeds): the ONLY numerical difference from v1
is compute placement — outer DDPG gradient updates on CUDA, frozen inner
rollout policy stays on CPU (per-step H2D transfer would dominate its tiny
3x256 forward pass). CUDA/CPU bit-identity is NOT claimed; v1 stays frozen.

Frozen sources (`tools/outer_rl/*`, `tools/outer_horizon_study.py`,
`tools/outer_reward_study.py`, hashed benchmarks) are imported, never edited.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from benchmarks.bldc.run import ROOT, sha
from benchmarks.bldc.current_rl.repeat_study import read, save, now
from tools import outer_horizon_study as parent
from tools.outer_rl import study as original, evaluation

GAMMAS = parent.GAMMAS
FAILURE_PENALTY = parent.FAILURE_PENALTY
PROTOCOL = 'outer-discount-horizon-v2'


def cuda_learner(env, widths, seed, cfg):
    """Mirror of `original.learner` with config-driven device (cuda for outer)."""
    c = cfg['learner']
    return DDPG('MlpPolicy', env, seed=seed, device=c.get('device', 'cpu'), learning_rate=c['learning_rate'],
        gamma=c['gamma'], tau=c['tau'], buffer_size=c['buffer_size'], learning_starts=c['learning_starts'],
        batch_size=c['batch_size'], train_freq=c['train_freq'], gradient_steps=c['gradient_steps'],
        action_noise=NormalActionNoise(np.zeros(1), np.ones(1) * c['noise_sigma']),
        policy_kwargs=dict(net_arch=dict(pi=widths, qf=c['critic_widths']), activation_fn=torch.nn.ReLU))


def prepare(output, inner, v1ref, budget=250000, seeds=(18, 19), validation_every=25000,
            device='cuda', max_workers=4, strict=True):
    """Build v2 children directly from the frozen inner (no reward-parent needed).

    strict=True asserts each fresh child config equals the matching v1 child
    config on disk except `learner.device`/`interpretation`, proving the rerun
    is matched except compute placement and fresh seeds.
    """
    output, inner, v1ref = Path(output).resolve(), Path(inner).resolve(), Path(v1ref).resolve()
    if budget <= 0 or validation_every <= 0 or budget % validation_every:
        raise ValueError('Budget must be positive multiple of validation interval')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Invalid seeds')
    output.mkdir(parents=True, exist_ok=False)
    label = '128x2'
    records, children = [], {}
    inner_manifest = read(inner / 'manifest.json')
    for group in ('g0995', 'g0999'):
        child = output / group
        original.prepare(child, inner, budget, seeds, [label], validation_every)
        cfg = read(child / 'config.json')
        cfg['reward'].update(speed_shape='l1', failure=FAILURE_PENALTY)
        cfg['learner']['gamma'] = GAMMAS[group]
        cfg['learner']['device'] = device
        cfg['interpretation'] = ('Paired discount-horizon experiment; L1 tracking and shared -5200 failure penalty; '
                                 'only gamma differs; v2 compute split: outer updates on %s, frozen inner rollouts on cpu' % device)
        if strict:
            ref = read(v1ref / group / 'config.json')
            mine = copy.deepcopy(cfg)
            mine['learner'].pop('device')
            mine.pop('interpretation')
            ref = copy.deepcopy(ref)
            ref.pop('interpretation')
            if mine != ref:
                raise ValueError('Fresh v2 config does not match v1 by value')
        save(child / 'config.json', cfg)
        protocol = read(child / 'protocol.json')
        protocol['config'] = cfg
        protocol['config_sha256'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        protocol['frozen_files']['config.json'] = sha(child / 'config.json')
        protocol['source_sha256'][str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
        protocol['horizon_group'] = group
        protocol['dispatch'] = 'tools.outer_horizon_v2 run; do not call original/parent CLI'
        save(child / 'protocol.json', protocol)
        (child / 'protocol.sha256').write_text(sha(child / 'protocol.json') + '\n')
        children[group] = sha(child / 'protocol.json')
        records.extend(dict(group=group, label=label, seed=s, arm=f'{label}-seed{s}') for s in seeds)
    p = dict(protocol=PROTOCOL, created_utc=now(), inner=str(inner), v1ref=str(v1ref),
        inner_manifest_sha256=sha(inner / 'manifest.json'),
        inner_model_sha256=inner_manifest.get('model_sha256'),
        children=children, records=records,
        seeds=list(seeds), label=label, budget=budget, max_workers=max_workers,
        max_outer_actions=len(records) * budget, max_training_physics_steps=10 * len(records) * budget,
        device=dict(learner=device, inner='cpu'), source_sha256=sha(Path(__file__)),
        selection='Failed seeds, incomplete cases, violations, median speed RMSE, gamma .995 before .999 on exact tie',
        hypothesis=('Longer discount horizon may reduce sustained offsets; efficacy unproven. '
                    'v2 reruns the v1 comparison (value-matched except compute placement and fresh seeds 18/19) '
                    'with outer updates on CUDA; no CUDA/CPU bit-identity claimed'),
        compute='RTX 3060 12GB, torch 2.13.0+cu130; outer learner device=%s, inner device=cpu' % device,
        lineage=('Value-matched to stopped outer-horizon-v1 child configs (minus device); '
                 'v1 reward-parent files unavailable on this host, L1/-5200/gamma asserted by value'),
        final_evaluation=False)
    save(output / 'protocol.json', p)
    (output / 'protocol.sha256').write_text(sha(output / 'protocol.json') + '\n')
    original.atomic(output / 'status.json', dict(status='prepared'))
    return p


def verify(output):
    output = Path(output)
    p = read(output / 'protocol.json')
    if sha(output / 'protocol.json') != (output / 'protocol.sha256').read_text().strip():
        raise ValueError('Changed v2 protocol')
    if p.get('protocol') != PROTOCOL:
        raise ValueError('Not a v2 protocol')
    if sha(Path(__file__)) != p['source_sha256']:
        raise ValueError('Changed v2 source')
    if sha(Path(p['inner']) / 'manifest.json') != p['inner_manifest_sha256']:
        raise ValueError('Changed frozen inner')
    expected = {(group, s) for group in ('g0995', 'g0999') for s in p['seeds']}
    if len(p['records']) != len(expected) or {(r['group'], r['seed']) for r in p['records']} != expected:
        raise ValueError('Invalid v2 queue')
    if set(p['children']) != set(GAMMAS):
        raise ValueError('Incomplete v2 groups')
    if any(r['label'] != p['label'] or r['arm'] != f"{p['label']}-seed{r['seed']}" for r in p['records']):
        raise ValueError('Invalid job identity')
    configs = {}
    for group, digest in p['children'].items():
        if sha(output / group / 'protocol.json') != digest:
            raise ValueError('Changed v2 child protocol')
        original.verify(output / group)
        c = read(output / group / 'config.json')
        if c['learner'].pop('gamma') != GAMMAS[group]:
            raise ValueError('Wrong gamma dispatch')
        if c['learner'].get('device') != p['device']['learner']:
            raise ValueError('Wrong learner device')
        if c['reward']['speed_shape'] != 'l1' or c['reward']['failure'] != FAILURE_PENALTY:
            raise ValueError('Wrong shared reward')
        configs[group] = c
    if configs['g0995'] != configs['g0999']:
        raise ValueError('Uncontrolled configuration difference')
    return p


def run(output, group, seed):
    output = Path(output)
    p = verify(output)
    if (group, seed) not in [(r['group'], r['seed']) for r in p['records']]:
        raise ValueError('Undeclared job')
    with parent.environment(group), patch.object(original, 'learner', cuda_learner):
        original.run_arm(output / group, p['label'], seed)


def execute(output, workers=None):
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    output = Path(output).resolve()
    p = verify(output)
    workers = workers or p.get('max_workers', 4)
    with (output / 'queue_started.json').open('x') as f:
        json.dump(dict(started_utc=now()), f)
    (output / 'logs').mkdir(exist_ok=False)
    original.atomic(output / 'status.json', dict(status='preparing_baselines'))
    try:
        original.torch.set_num_threads(1)
        original.torch.use_deterministic_algorithms(True)
        for group in ('g0995', 'g0999'):
            child = output / group
            c = read(child / 'config.json')
            inner = DDPG.load(child / 'inner_model.zip', device='cpu')
            with parent.environment(group):
                evaluation.baselines(c['plant'], c['inner_study'], inner, c['validation_cases'], child / 'baselines')
        original.atomic(output / 'status.json', dict(status='running'))
        def job(r):
            started = now()
            try:
                with (output / 'logs' / f"{r['group']}-{r['arm']}.log").open('x') as log:
                    process = subprocess.run([sys.executable, '-m', 'tools.outer_horizon_v2', 'run', '--output', str(output),
                        '--group', r['group'], '--seed', str(r['seed'])], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                return dict(**r, returncode=process.returncode, started_utc=started, completed_utc=now())
            except Exception as exc:
                return dict(**r, returncode=-1, failure=f'{type(exc).__name__}: {exc}')
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(job, r) for r in p['records']]
            for future in as_completed(futures):
                results.append(future.result())
                original.atomic(output / 'queue_status.json', dict(completed_jobs=len(results), total_jobs=len(p['records']), results=results))
                print(json.dumps(results[-1]), flush=True)
        save(output / 'run_summary.json', results)
        if any(r['returncode'] for r in results):
            raise RuntimeError('Failed job; no aggregate selection')
        select(output)
        original.atomic(output / 'status.json', dict(status='completed', completed_utc=now()))
    except BaseException as exc:
        original.atomic(output / 'status.json', dict(status='failed', failure=f'{type(exc).__name__}: {exc}', failed_utc=now()))
        raise


def select(output):
    output = Path(output)
    p = verify(output)
    if (output / 'selection.json').exists():
        raise FileExistsError('Selection already committed')
    queue = read(output / 'run_summary.json')
    if (len(queue) != len(p['records'])
            or {(r['group'], r['seed']) for r in queue} != {(r['group'], r['seed']) for r in p['records']}
            or any(r['returncode'] for r in queue)):
        raise RuntimeError('Incomplete queue')
    rows, scores = [], {}
    for i, group in enumerate(('g0995', 'g0999')):
        child = output / group
        save(child / 'run_summary.json', [r for r in queue if r['group'] == group])
        original.select(child)
        selected = read(child / 'selection.json')
        for row in selected['rows']:
            rows.append(dict(group=group, **row))
        scores[group] = selected['scores'][p['label']][:-1] + [i]
        original.atomic(child / 'status.json', dict(status='completed', completed_utc=now()))
    result = dict(rows=rows, scores=scores, winner=min(scores, key=scores.get), final_qualification=False,
                  all_winner_seeds_pass_validation=min(scores.values())[0] == 0)
    save(output / 'selection.json', result)
    (output / 'selection.sha256').write_text(sha(output / 'selection.json') + '\n')
    lines = ['# Outer discount-horizon v2 (CUDA) development comparison', '',
             f"Validation-ranked horizon: {result['winner']}. No final qualification.", '',
             '| Horizon | Seed | Selected step | Cases passed | Speed RMSE rad/s |', '|---|---:|---:|---:|---:|']
    for row in rows:
        s = row['selected']
        lines.append(f"| {row['group']} | {row['seed']} | {s['step']} | {s['passed_cases']}/{s['case_count']} | {s['score'][2]:.6f} |")
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['prepare', 'run', 'execute'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inner', type=Path, default=Path('results/bldc/outer-inner-freeze-v1'))
    ap.add_argument('--v1ref', type=Path, default=Path('results/bldc/outer-horizon-v1'))
    ap.add_argument('--group', choices=['g0995', 'g0999'])
    ap.add_argument('--seed', type=int)
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare(a.output, a.inner, a.v1ref)
    elif a.command == 'run':
        run(a.output, a.group, a.seed)
    else:
        execute(a.output, a.workers)


if __name__ == '__main__':
    main()
