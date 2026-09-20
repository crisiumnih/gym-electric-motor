"""Outer memory-gain fix screen (v2 contract), CUDA updates.

Mechanism (measured on horizon-v2 traces, see docs/bldc_outer_fix_v1_protocol.md):
the v1 bounded error memory accumulates 0.002*en per outer action, so over a
full episode with persistent bias |z| never exceeds 0.02 and its cost <=2e-5:
the actor gets no usable integral signal. With no integral authority its only
tools are P-kicks (relay) or nothing (lazy) -- one root, two symptoms.

This screen changes ONE variable: 10x memory gain (divisor 0.05 vs 0.5).
Reward cost is rescaled 0.5 -> 0.005 so pre-clip cost is IDENTICAL for
identical error history (z_new = 10*z_old => 0.005*100*z_old^2 = 0.5*z_old^2);
only the observation carries 10x signal. Gamma/L1/failure/widths/budget/
device match horizon-v2 g0995; fresh seeds 24/25.

Frozen sources are imported, never edited. The step below is a full explicit
copy of OuterEnv.step with exactly three documented deltas.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
from benchmarks.bldc.environment import schedule_value
from benchmarks.bldc.run import ROOT, sha
from benchmarks.bldc.current_rl.repeat_study import read, save, now
from tools.outer_rl import study as original, evaluation
from tools.outer_rl.environment import OuterEnv
from tools import outer_horizon_study as horizon
from tools.outer_horizon_v2 import cuda_learner

PROTOCOL = 'outer-memory-gain-v1'
CONTRACT = 'bldc-outer-speed-v2'
LABEL = '128x2'
GAMMA = 0.995
FAILURE_PENALTY = -5200.0
MEMORY_DIVISOR = 0.05  # DELTA 3a (was 0.5): 10x integrator gain
MEMORY_COST = 0.005    # DELTA 3b (was 0.5): identical pre-clip cost, see docstring


class FixV1Env(horizon.HorizonEnv):
    """v2-contract outer env. Step = OuterEnv.step verbatim except:
    DELTA 1 (from AbsoluteTrackingEnv.step): L1 absolute-tracking adjustment.
    DELTA 2 (from HorizonEnv.step): -5200 failure override on termination.
    DELTA 3 (new): 10x memory gain + rescaled memory cost (see docstring).
    Observation order/scales, timing, limits, memory/reset semantics unchanged.
    """
    def step(self, action):
        if self.done:
            raise RuntimeError('reset required after episode completion')
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.shape != (1,) or not np.isfinite(a).all():
            raise ValueError('One finite scalar action required')
        self.last_rows = []
        command = 1.5 * float(np.clip(a[0], -1, 1))
        pre, reference = self.state_reference()
        z = self.z
        delta = abs(command - self.previous_command) / 3.
        rows = []
        terms = []
        terminated = False
        truncated = False
        failure = None
        duration = round(self.case['duration_s'] / self.tau_i)
        for _ in range(min(self.hold_steps, duration - self.physics_steps)):
            before, ref = self.state_reference()
            self.load.disturbance_nm = schedule_value(self.case['disturbance'], self.physics_steps, self.tau_i)
            voltage = np.asarray(self.current.act_current(before, [0., command], self.tau_i), dtype=float)
            if voltage.shape != (2,) or not np.isfinite(voltage).all():
                raise RuntimeError('Invalid inner action')
            self.observation, _, native_term, native_trunc, _ = self.env.step(voltage)
            after, _ = self.state_reference()
            if not np.isfinite(list(after.values())).all():
                raise RuntimeError('Nonfinite physical state')
            self.physics_steps += 1
            phase_peak = max(abs(after[k]) for k in ['i_a', 'i_b', 'i_c'])
            terminated = bool(native_term or phase_peak > 4.)
            truncated = bool(native_trunc and not terminated)
            failure = ('phase_current_trip' if phase_peak > 4 else 'native_constraint') if terminated else ('native_truncation' if truncated else None)
            components = dict(speed=4. * float(np.clip((ref - after['omega']) / 25, -1, 1)) ** 2,
                current=.5 * float(np.clip(np.hypot(after['i_sd'], after['i_sq']) / 4, 0, 1)) ** 2,
                command_delta=.1 * float(np.clip(delta, 0, 1)) ** 2, memory=MEMORY_COST * z * z,
                torque_delta=.1 * float(np.clip(abs(after['torque'] - before['torque']) / .1, 0, 1)) ** 2)
            terms.append(components)
            indices = [self.names.index(n) for n in ['i_a', 'i_b', 'i_c']]
            row = dict(after, **self.current.diagnostics, time_s=self.physics_steps * self.tau_i,
                reference_rad_s=ref, omega_before=float(before['omega']), disturbance_nm=float(self.load.disturbance_nm),
                requested_d=float(voltage[0]), requested_q=float(voltage[1]), action_d=float(voltage[0]), action_q=float(voltage[1]),
                action_clipped=0., action_at_limit=float(np.linalg.norm(voltage) >= self.plant['action_norm_limit'] - 1e-12),
                outer_action=float(a[0]), outer_command_a=command, outer_action_clipped=float(abs(a[0]) > 1),
                reward=-sum(components.values()), phase_current_ratio=phase_peak / 4., current_constraint_ratio=float(np.sum(np.asarray(self.observation[0])[indices] ** 2)),
                terminated=float(terminated), truncated=float(truncated))
            rows.append(row)
            self.last_rows = rows
            if terminated or truncated:
                break
        if not rows:
            raise RuntimeError('No inner samples in outer transition')
        self.outer_actions += 1
        self.z = float(np.clip(z + len(rows) * self.tau_i * np.clip((reference - pre['omega']) / 25, -1, 1) / MEMORY_DIVISOR, -1, 1))
        self.previous_command = command
        truncated = bool(truncated or (not terminated and self.physics_steps >= duration))
        self.done = terminated or truncated
        reward = FAILURE_PENALTY if terminated else -float(np.mean([sum(c.values()) for c in terms]))
        mean_terms = {k: float(np.mean([c[k] for c in terms])) for k in terms[0]}
        mean_terms['failure_penalty'] = FAILURE_PENALTY if terminated else 0.
        # DELTA 1: L1 absolute-tracking adjustment (AbsoluteTrackingEnv.step lines 23-29).
        l1_delta, l1_cost = [], []
        for row in rows:
            en = float(np.clip((row['reference_rad_s'] - row['omega']) / 25., -1., 1.))
            difference = 4. * (abs(en) - en ** 2)
            row['reward'] -= difference
            l1_delta.append(difference)
            l1_cost.append(4. * abs(en))
        mean_terms['speed'] = float(np.mean(l1_cost))
        if not terminated:
            reward -= float(np.mean(l1_delta))
        info = dict(inner_steps=len(rows), physics_steps=self.physics_steps, outer_actions=self.outer_actions,
            failure=failure, rows=rows, reward_terms=mean_terms, scenario=self.case['name'],
            time_limit=bool(truncated and not failure and self.physics_steps >= duration))
        return self._obs(), float(reward), bool(terminated), bool(truncated), info


@contextmanager
def environment():
    with patch.object(original, 'OuterEnv', FixV1Env), patch.object(evaluation, 'OuterEnv', FixV1Env):
        yield


def prepare(output, inner, budget=250000, seeds=(24, 25), validation_every=25000,
            device='cuda', max_workers=2):
    output, inner = Path(output).resolve(), Path(inner).resolve()
    if budget <= 0 or validation_every <= 0 or budget % validation_every:
        raise ValueError('Budget must be positive multiple of validation interval')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Invalid seeds')
    output.mkdir(parents=True, exist_ok=False)
    child = output / 'fix'
    original.prepare(child, inner, budget, seeds, [LABEL], validation_every)
    cfg = read(child / 'config.json')
    cfg['reward'].update(speed_shape='l1', failure=FAILURE_PENALTY)
    cfg['learner'].update(gamma=GAMMA, device=device)
    cfg['contract'].update(version=CONTRACT, memory_time_s=MEMORY_DIVISOR)
    cfg['interpretation'] = ('Memory-gain fix screen; v2 contract (10x integrator gain, rescaled memory cost); '
                             'else gamma .995/L1/-5200/2x128 matched to horizon-v2 g0995')
    save(child / 'config.json', cfg)
    protocol = read(child / 'protocol.json')
    protocol['config'] = cfg
    protocol['config_sha256'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    protocol['frozen_files']['config.json'] = sha(child / 'config.json')
    protocol['source_sha256'][str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    protocol['dispatch'] = 'tools.outer_fix_v1 run; do not call original CLI'
    save(child / 'protocol.json', protocol)
    (child / 'protocol.sha256').write_text(sha(child / 'protocol.json') + '\n')
    m = read(inner / 'manifest.json')
    p = dict(protocol=PROTOCOL, created_utc=now(), inner=str(inner),
        inner_manifest_sha256=sha(inner / 'manifest.json'), inner_model_sha256=m.get('model_sha256'),
        child=sha(child / 'protocol.json'), records=[dict(label=LABEL, seed=s, arm=f'{LABEL}-seed{s}') for s in seeds],
        seeds=list(seeds), label=LABEL, budget=budget, max_workers=max_workers,
        max_outer_actions=len(seeds) * budget, max_training_physics_steps=10 * len(seeds) * budget,
        device=dict(learner=device, inner='cpu'), source_sha256=sha(Path(__file__)),
        selection='Failed seeds, incomplete cases, gate violations, median speed RMSE',
        hypothesis=('10x memory gain gives the actor integral authority the v1 channel provably lacks '
                    '(|z|<=0.02 over full episodes); relay/lazy bistability should collapse toward tracking. '
                    'Reward rescale keeps pre-clip cost identical, so any change is the observation, not the objective.'),
        contract=dict(version=CONTRACT, memory_divisor=MEMORY_DIVISOR, memory_cost=MEMORY_COST,
                      changes='memory gain x10, memory cost /100; order/scales/timing/limits/reset unchanged'),
        final_evaluation=False)
    save(output / 'protocol.json', p)
    (output / 'protocol.sha256').write_text(sha(output / 'protocol.json') + '\n')
    original.atomic(output / 'status.json', dict(status='prepared'))
    return p


def verify(output):
    output = Path(output)
    p = read(output / 'protocol.json')
    if sha(output / 'protocol.json') != (output / 'protocol.sha256').read_text().strip():
        raise ValueError('Changed fix protocol')
    if p.get('protocol') != PROTOCOL:
        raise ValueError('Not a fix protocol')
    if sha(Path(__file__)) != p['source_sha256']:
        raise ValueError('Changed fix source')
    if sha(Path(p['inner']) / 'manifest.json') != p['inner_manifest_sha256']:
        raise ValueError('Changed frozen inner')
    if sha(output / 'fix' / 'protocol.json') != p['child']:
        raise ValueError('Changed fix child protocol')
    original.verify(output / 'fix')
    c = read(output / 'fix' / 'config.json')
    if c['contract'].get('version') != CONTRACT or c['contract'].get('memory_time_s') != MEMORY_DIVISOR:
        raise ValueError('Wrong v2 contract')
    if c['learner'].get('device') != p['device']['learner'] or c['learner'].get('gamma') != GAMMA:
        raise ValueError('Wrong learner dispatch')
    if c['reward']['speed_shape'] != 'l1' or c['reward']['failure'] != FAILURE_PENALTY:
        raise ValueError('Wrong shared reward')
    expected = {(LABEL, s) for s in p['seeds']}
    if len(p['records']) != len(expected) or {(r['label'], r['seed']) for r in p['records']} != expected:
        raise ValueError('Invalid fix queue')
    return p


def run(output, label, seed):
    output = Path(output)
    p = verify(output)
    if (label, seed) not in [(r['label'], r['seed']) for r in p['records']]:
        raise ValueError('Undeclared job')
    import tools.outer_rl.study as orig
    with patch.object(orig, 'OuterEnv', FixV1Env), patch.object(orig, 'learner',
            lambda env, widths, seed, cfg: cuda_learner(env, widths, seed, cfg)):
        orig.run_arm(output / 'fix', label, seed)


def execute(output, workers=None):
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from stable_baselines3 import DDPG
    output = Path(output).resolve()
    p = verify(output)
    workers = workers or p.get('max_workers', 2)
    with (output / 'queue_started.json').open('x') as f:
        json.dump(dict(started_utc=now()), f)
    (output / 'logs').mkdir(exist_ok=False)
    original.atomic(output / 'status.json', dict(status='preparing_baselines'))
    try:
        original.torch.set_num_threads(1)
        original.torch.use_deterministic_algorithms(True)
        cfg = read(output / 'fix' / 'config.json')
        inner = DDPG.load(output / 'fix' / 'inner_model.zip', device='cpu')
        with environment():
            evaluation.baselines(cfg['plant'], cfg['inner_study'], inner, cfg['validation_cases'], output / 'fix' / 'baselines')
        original.atomic(output / 'status.json', dict(status='running'))

        def job(r):
            started = now()
            try:
                with (output / 'logs' / f"{r['arm']}.log").open('x') as log:
                    process = subprocess.run([sys.executable, '-m', 'tools.outer_fix_v1', 'run', '--output', str(output),
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
            or {(r['label'], r['seed']) for r in queue} != {(r['label'], r['seed']) for r in p['records']}
            or any(r['returncode'] for r in queue)):
        raise RuntimeError('Incomplete queue')
    original.select(output / 'fix')
    selected = read(output / 'fix' / 'selection.json')
    rows = [r for r in selected['rows']]
    result = dict(rows=rows, scores=selected.get('scores', {}), winner=LABEL,
                  final_qualification=False,
                  all_winner_seeds_pass_validation=all(r['selected']['passed_cases'] == r['selected']['case_count'] for r in rows))
    save(output / 'selection.json', result)
    (output / 'selection.sha256').write_text(sha(output / 'selection.json') + '\n')
    lines = ['# Outer memory-gain fix development comparison', '',
             'Seeds 24/25, v2 contract. No final qualification.', '',
             '| Seed | Selected step | Cases passed | Speed RMSE rad/s |', '|---|---:|---:|---:|']
    for row in rows:
        s = row['selected']
        lines.append(f"| {row['seed']} | {s['step']} | {s['passed_cases']}/{s['case_count']} | {s['score'][2]:.6f} |")
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return result


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
