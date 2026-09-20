"""Effort-penalty ablation on Warp training + GEM validation (new lineage).

g0995 values everywhere except the effort multiplier on command_delta and
torque_delta terms: x1 / x5 / x10. Hypothesis: x1 learns rail-to-rail relay
(observed); pricing effort kills the relay and unlocks settling gates.
Single variable; 2x128 only; fresh seeds 22/23; 150k budget, 25k validation.

Each effort child is a full ceiling-shaped root (verified by ceiling.verify);
the effort root only aggregates. Lineage: outer-warp-effort-v1.
Frozen GEM sources imported, never edited.
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from benchmarks.bldc.run import ROOT, sha
from benchmarks.bldc.current_rl.repeat_study import read, save, now
from tools.outer_rl import study as original, evaluation
from tools import outer_horizon_study as horizon
from tools import outer_warp_ceiling as ceiling

PROTOCOL = 'outer-warp-effort-v1'
EFFORTS = (1.0, 5.0, 10.0)
LABEL = '128x2'


def prepare(output, inner, budget=150000, seeds=(22, 23), efforts=EFFORTS,
            validation_every=25000, max_workers=4):
    output, inner = Path(output).resolve(), Path(inner).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Invalid seeds')
    children, records = {}, []
    for effort in efforts:
        child = output / f'e{effort:g}'
        ceiling.prepare(child, inner, budget=budget, seeds=list(seeds), widths=(LABEL,),
                        validation_every=validation_every, max_workers=max_workers,
                        reward_extra={'effort_scale': effort},
                        interpretation_note=f'; effort ablation arm x{effort:g}')
        children[f'e{effort:g}'] = sha(child / 'protocol.json')
        records.extend(dict(effort=effort, label=LABEL, seed=s, arm=f'e{effort:g}/{LABEL}-seed{s}')
                       for s in seeds)
    m = read(inner / 'manifest.json')
    p = dict(protocol=PROTOCOL, created_utc=now(), inner=str(inner),
        inner_manifest_sha256=sha(inner / 'manifest.json'),
        inner_model_sha256=m.get('model_sha256'), children=children, records=records,
        seeds=list(seeds), efforts=list(efforts), label=LABEL, budget=budget, max_workers=max_workers,
        max_outer_actions=len(records) * budget, max_training_physics_steps=10 * len(records) * budget,
        device=dict(learner='cuda', inner='cuda', physics='warp-cuda', validation='gem-cpu'),
        source_sha256=sha(Path(__file__)),
        selection='Failed efforts, incomplete cases, gate violations, median speed RMSE; relay judged from validation command traces',
        hypothesis=('x1 reproduces relay (command rail-to-rail); x5/x10 make relay uneconomical, '
                    'restoring settling/tail gates. Efficacy unproven until run.'),
        lineage='New Warp-training lineage; effort is the only difference between arms',
        final_evaluation=False)
    save(output / 'protocol.json', p)
    (output / 'protocol.sha256').write_text(sha(output / 'protocol.json') + '\n')
    original.atomic(output / 'status.json', dict(status='prepared'))
    return p


def verify(output):
    output = Path(output)
    p = read(output / 'protocol.json')
    if sha(output / 'protocol.json') != (output / 'protocol.sha256').read_text().strip():
        raise ValueError('Changed effort protocol')
    if p.get('protocol') != PROTOCOL:
        raise ValueError('Not an effort protocol')
    if sha(Path(__file__)) != p['source_sha256']:
        raise ValueError('Changed effort source')
    if sha(Path(p['inner']) / 'manifest.json') != p['inner_manifest_sha256']:
        raise ValueError('Changed frozen inner')
    expected = {(e, s) for e in p['efforts'] for s in p['seeds']}
    if len(p['records']) != len(expected) or {(r['effort'], r['seed']) for r in p['records']} != expected:
        raise ValueError('Invalid effort queue')
    if set(p['children']) != {f'e{e:g}' for e in p['efforts']}:
        raise ValueError('Incomplete effort groups')
    configs = {}
    for group, digest in p['children'].items():
        if sha(output / group / 'protocol.json') != digest:
            raise ValueError('Changed effort child protocol')
        ceiling.verify(output / group)
        c = read(output / group / 'config.json')
        if c['reward'].pop('effort_scale') != float(group[1:]):
            raise ValueError('Wrong effort dispatch')
        configs[group] = c
    base = next(iter(configs.values()))
    if any(c != base for c in configs.values()):
        raise ValueError('Uncontrolled configuration difference')
    return p


def run(output, effort, seed):
    output = Path(output)
    p = verify(output)
    if (effort, seed) not in [(r['effort'], r['seed']) for r in p['records']]:
        raise ValueError('Undeclared job')
    ceiling.run_arm(output / f'e{effort:g}', p['label'], seed)


def execute(output, workers=None):
    output = Path(output).resolve()
    p = verify(output)
    workers = workers or p.get('max_workers', 4)
    with (output / 'queue_started.json').open('x') as f:
        json.dump(dict(started_utc=now()), f)
    (output / 'logs').mkdir(exist_ok=False)
    original.atomic(output / 'status.json', dict(status='preparing_baselines'))
    try:
        import torch
        from stable_baselines3 import DDPG
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        for group in p['children']:
            child = output / group
            c = read(child / 'config.json')
            inner = DDPG.load(child / 'inner_model.zip', device='cpu')
            with horizon.environment('g0995'):
                evaluation.baselines(c['plant'], c['inner_study'], inner,
                                     c['validation_cases'], child / 'baselines')
        original.atomic(output / 'status.json', dict(status='running'))

        def job(r):
            started = now()
            try:
                with (output / 'logs' / f"e{r['effort']:g}-{r['arm'].replace('/', '_')}.log").open('x') as log:
                    process = subprocess.run(
                        [sys.executable, '-m', 'tools.outer_warp_effort', 'run', '--output', str(output),
                         '--effort', str(r['effort']), '--seed', str(r['seed'])],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                return dict(**r, returncode=process.returncode, started_utc=started, completed_utc=now())
            except Exception as exc:
                return dict(**r, returncode=-1, failure=f'{type(exc).__name__}: {exc}')
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(job, r) for r in p['records']]
            for future in as_completed(futures):
                results.append(future.result())
                original.atomic(output / 'queue_status.json', dict(completed_jobs=len(results),
                       total_jobs=len(p['records']), results=results))
                print(json.dumps(results[-1]), flush=True)
        save(output / 'run_summary.json', results)
        if any(r['returncode'] for r in results):
            raise RuntimeError('Failed job; no aggregate selection')
        select(output)
        original.atomic(output / 'status.json', dict(status='completed', completed_utc=now()))
    except BaseException as exc:
        original.atomic(output / 'status.json', dict(status='failed',
               failure=f'{type(exc).__name__}: {exc}', failed_utc=now()))
        raise


def select(output):
    output = Path(output)
    p = verify(output)
    if (output / 'selection.json').exists():
        raise FileExistsError('Selection already committed')
    queue = read(output / 'run_summary.json')
    if (len(queue) != len(p['records'])
            or {(r['effort'], r['seed']) for r in queue} != {(r['effort'], r['seed']) for r in p['records']}
            or any(r['returncode'] for r in queue)):
        raise RuntimeError('Incomplete queue')
    rows = []
    for r in p['records']:
        group = f'e{r["effort"]:g}'
        out = output / group / f'{r["label"]}-seed{r["seed"]}'
        m = read(out / 'manifest.json')
        if m['status'] != 'completed':
            raise RuntimeError('Incomplete job')
        for rel, digest in m['artifacts_sha256'].items():
            if sha(out / rel) != digest:
                raise ValueError('Changed training artifact')
        if sha(out / 'best_model.zip') != m['selected']['sha256']:
            raise ValueError('Changed selected model')
        rows.append(dict(effort=r['effort'], label=r['label'], seed=r['seed'],
                         arm=r['arm'], selected=m['selected'],
                         outer_actions=m['outer_actions'], failed_episodes=m['failed_episodes']))
    by_effort = {}
    for row in rows:
        by_effort.setdefault(row['effort'], []).append(row)
    scores = {e: [min(1, 6 - max(r['selected']['passed_cases'] for r in rs)), 0,
                  float(np.median([r['selected']['score'][2] for r in rs]))]
              for e, rs in by_effort.items()}
    result = dict(rows=rows, scores={str(k): v for k, v in scores.items()},
                  winner=min(scores, key=scores.get), final_qualification=False,
                  all_winner_seeds_pass_validation=all(
                      r['selected']['passed_cases'] == r['selected']['case_count']
                      for r in rows if r['effort'] == min(scores, key=scores.get)))
    save(output / 'selection.json', result)
    (output / 'selection.sha256').write_text(sha(output / 'selection.json') + '\n')
    lines = ['# Outer warp effort development comparison', '',
             f"Validation-ranked effort: x{result['winner']:g}. No final qualification.", '',
             '| Effort | Seed | Selected step | Cases passed | Speed RMSE rad/s |', '|---|---:|---:|---:|---:|']
    for row in rows:
        s = row['selected']
        lines.append(f"| x{row['effort']:g} | {row['seed']} | {s['step']} | {s['passed_cases']}/{s['case_count']} | {s['score'][2]:.6f} |")
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['prepare', 'run', 'execute', 'select'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inner', type=Path, default=Path('results/bldc/outer-inner-freeze-v1'))
    ap.add_argument('--effort', type=float)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare(a.output, a.inner)
    elif a.command == 'run':
        run(a.output, a.effort, a.seed)
    elif a.command == 'execute':
        execute(a.output, a.workers)
    else:
        select(a.output)


if __name__ == '__main__':
    main()
