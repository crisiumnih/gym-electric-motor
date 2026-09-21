"""Repaired direct baseline + IASA stage (Warp trains, GEM judges).

Two arms, one root (read post-hoc as stages):
- direct-td3: TD3 scalar 7-obs v2-memory L1/-5200 (repair-effect control:
  per-lane coverage, TimeLimit fix, 8 updates/transition).
- iasa-td3: TD3 two-output 19-obs v3 IASA, L1-family reward without z-memory
  term (memory channel does not exist in v3; candidate reward is next stage).
Shared: gamma .995, 2x128/2x256, 250k/env x8, 25k/env validation, frozen
inner, fresh seeds 32-35, gradient_steps=8 (one update per transition).
Validation AND baselines run the contract-dispatched GEM adapter, so policies
are scored under their own contract. New lineage, no CPU-run comparison.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from benchmarks.bldc.run import ROOT, sha
from benchmarks.bldc.current_rl.repeat_study import read, save, now
from tools.outer_rl import study as original, evaluation
from tools.outer_env_contract import env_for_contract, V1, V2, V3

PROTOCOL = 'outer-iasa-v1'
ARMS = {
    'direct-td3': dict(contract=V2, obs=7, outputs=1, memory=(0.05, 0.005)),
    'iasa-td3': dict(contract=V3, obs=19, outputs=2, memory=None),
}
SEEDS = (32, 33, 34, 35)
BUDGET = 250000
VALID_EVERY = 25000
N_ENVS = 8
LABEL_ARCH = {'direct-td3': [128, 128], 'iasa-td3': [128, 128]}


@contextmanager
def contract_env(version):
    with patch.object(evaluation, 'OuterEnv', lambda *a, **k: env_for_contract(version, *a, **k)):
        yield


def learner(env, widths, seed, cfg):
    c = cfg['learner']
    return TD3('MlpPolicy', env, seed=seed, device='cuda', learning_rate=c['learning_rate'],
        gamma=c['gamma'], tau=c['tau'], buffer_size=c['buffer_size'], learning_starts=c['learning_starts'],
        batch_size=c['batch_size'], train_freq=c['train_freq'], gradient_steps=c['gradient_steps'],
        action_noise=NormalActionNoise(np.zeros(env.action_space.shape[0]),
                                       np.ones(env.action_space.shape[0]) * c['noise_sigma']),
        policy_kwargs=dict(net_arch=dict(pi=widths, qf=c['critic_widths']), activation_fn=torch.nn.ReLU))


class Progress(BaseCallback):
    def __init__(self, output, root, config, inner, version):
        super().__init__()
        self.out = output
        self.root = root
        self.c = config
        self.inner = inner
        self.version = version
        self.best = None
        self.physics = 0
        self.episodes = 0
        self.failures = 0
        self.started = time.perf_counter()

    def _on_step(self):
        infos = self.locals['infos']
        dones = self.locals['dones']
        self.physics += sum(i['physics_steps'] for i in infos)
        self.failures += sum(1 for i in infos if i.get('failure'))
        self.episodes += int(np.asarray(dones).sum())
        if self.num_timesteps % self.c['validation_every'] == 0:
            with contract_env(self.version):
                rows = evaluation.compare(self.model, self.c['plant'], self.c['inner_study'], self.inner,
                    self.c['validation_cases'], self.out / 'validation' / str(self.num_timesteps),
                    baseline_directory=self.root / self.c['baseline_dir'])
            value = original.score(rows)
            if not np.isfinite(value[-1]):
                raise RuntimeError('Nonfinite or missing validation RMSE')
            if self.best is None or value < self.best['score']:
                self.model.save(self.out / 'best_model')
                self.best = dict(step=self.num_timesteps, score=value, sha256=sha(self.out / 'best_model.zip'),
                    passed_cases=sum(r['passed'] for r in rows), case_count=len(rows))
                original.atomic(self.out / 'selection.json', self.best)
        if self.num_timesteps % 1000 == 0:
            original.atomic(self.out / 'status.json', dict(status='running', outer_actions=self.num_timesteps,
                physics_steps=self.physics, updates=self.model._n_updates, episodes=self.episodes,
                failed_episodes=self.failures, selected=self.best,
                elapsed_s=time.perf_counter() - self.started))
        return True


def prepare(output, inner, budget=BUDGET, seeds=SEEDS, validation_every=VALID_EVERY, max_workers=4, learner_extra=None):
    import shutil
    output, inner = Path(output).resolve(), Path(inner).resolve()
    seeds = list(seeds)
    if output.exists():
        raise FileExistsError(output)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Invalid seeds')
    if budget <= 0 or validation_every <= 0 or budget % validation_every:
        raise ValueError('Budget must be positive multiple of validation interval')
    m = read(inner / 'manifest.json')
    for name, key in [('inner_model.zip', 'model_sha256'), ('inner_config.json', 'config_sha256')]:
        if sha(inner / name) != m[key]:
            raise ValueError('Changed frozen inner')
    output.mkdir(parents=True)
    for name in ['inner_model.zip', 'inner_config.json', 'manifest.json']:
        shutil.copyfile(inner / name, output / ('inner_freeze.json' if name == 'manifest.json' else name))
    saved = read(output / 'inner_config.json')
    saved['plant']['controller']['current_reference_limit_a'] = saved['study']['reference_limit_a']
    from tools.outer_rl.study import training_cases, validation_cases
    cfgs = {}
    for arm, spec in ARMS.items():
        mem_div, mem_cost = spec['memory'] or (0.5, 0.5)
        learner = dict(algorithm='td3', gamma=.995, learning_rate=1e-4, batch_size=128, buffer_size=300000,
                       learning_starts=5000 * N_ENVS, tau=.005, train_freq=1, gradient_steps=N_ENVS,
                       noise_sigma=.05, critic_widths=[256, 256], device='cuda')
        learner.update(learner_extra or {})
        cfgs[arm] = dict(
            protocol='bldc-outer-iasa-v1', plant=saved['plant'], inner_study=saved['study'],
            training_cases=training_cases(), validation_cases=validation_cases(), test_cases=[],
            contract_version=spec['contract'], n_envs=N_ENVS,
            learner=learner,
            contract=dict(version=spec['contract'], memory_time_s=mem_div if spec['memory'] else None),
            reward=dict(shape='l1', failure=-5200.0, memory_divisor=mem_div, memory_cost=mem_cost),
            validation_every=validation_every * N_ENVS, total_timesteps=budget * N_ENVS,
            per_env_timesteps=budget, baseline_dir=f'baselines-{arm}',
            interpretation=f'IASA stage arm {arm}; Warp trains, contract GEM validates')
    save(output / 'configs.json', cfgs)
    import importlib.metadata
    from benchmarks.bldc.run import provenance
    p = provenance(cfgs)
    p['packages'].update({n: importlib.metadata.version(n) for n in ['torch', 'stable-baselines3']})
    for f in sorted((ROOT / 'tools/outer_rl').glob('*.py')):
        p['source_sha256'][str(f.relative_to(ROOT))] = sha(f)
    for rel in ['tools/outer_iasa_v1.py', 'tools/outer_env_contract.py',
                'tools/outer_warp_ceiling.py', 'tools/outer_horizon_study.py', 'tools/outer_reward_study.py']:
        p['source_sha256'][rel] = sha(ROOT / rel)
    vroom = Path('/home/sra/prajwal/vroom')
    for rel in ['rl/outer/contract.py', 'rl/outer/controller.py', 'rl/outer/reward.py',
                'warp_backend/rollout.py', 'warp_backend/backend.py', 'warp_backend/vecenv.py']:
        p['source_sha256']['vroom:' + rel] = sha(vroom / rel)
    p.update(protocol=PROTOCOL, seeds=seeds, arms=list(ARMS), max_workers=max_workers,
        budget=budget, n_envs=N_ENVS, max_outer_actions=len(ARMS) * len(seeds) * budget,
        max_training_physics_steps=10 * len(ARMS) * len(seeds) * budget,
        device=dict(learner='cuda', inner='cuda', physics='warp-cuda', validation='gem-cpu'),
        frozen_files={n: sha(output / n) for n in ['configs.json', 'inner_model.zip', 'inner_config.json', 'inner_freeze.json']},
        records=[dict(arm=arm, label=arm, seed=s, job=f'{arm}-seed{s}') for arm in ARMS for s in seeds],
        selection='Failed seeds, incomplete cases, gate violations, median speed RMSE per arm',
        initialization='SB3 default random actor/critics; empty replay; no PI labels or warm start',
        lineage='Repaired direct baseline + IASA action/obs stage; Warp trains, contract GEM validates',
        final_evaluation=False)
    save(output / 'protocol.json', p)
    (output / 'protocol.sha256').write_text(sha(output / 'protocol.json') + '\n')
    original.atomic(output / 'status.json', dict(status='prepared'))
    return p


def verify(output):
    output = Path(output)
    p = read(output / 'protocol.json')
    if sha(output / 'protocol.json') != (output / 'protocol.sha256').read_text().strip():
        raise ValueError('Changed IASA protocol')
    if p.get('protocol') != PROTOCOL:
        raise ValueError('Not an IASA protocol')
    import importlib.metadata
    for package, version in p['packages'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'Changed package {package}')
    for name, digest in p['frozen_files'].items():
        if sha(output / name) != digest:
            raise ValueError(f'Changed frozen input {name}')
    for rel, digest in p['source_sha256'].items():
        base = rel[6:] if rel.startswith('vroom:') else rel
        root = Path('/home/sra/prajwal/vroom') if rel.startswith('vroom:') else ROOT
        if sha(root / base) != digest:
            raise ValueError(f'Changed source {rel}')
    expected = {(arm, s) for arm in ARMS for s in p['seeds']}
    if len(p['records']) != len(expected) or {(r['arm'], r['seed']) for r in p['records']} != expected:
        raise ValueError('Invalid IASA queue')
    cfgs = read(output / 'configs.json')
    if set(cfgs) != set(ARMS):
        raise ValueError('Incomplete arm configs')
    for arm, spec in ARMS.items():
        c = cfgs[arm]
        if c['contract_version'] != spec['contract'] or c['learner'].get('algorithm', 'td3') != 'td3':
            raise ValueError(f'Wrong dispatch for {arm}')
        if spec['outputs'] == 2 and (c['contract'].get('version') != 'bldc-outer-speed-v3'):
            raise ValueError(f'Wrong v3 contract for {arm}')
    return p


def run_arm(root, arm, seed):
    sys.path.insert(0, '/home/sra/prajwal/vroom')
    from warp_backend.vecenv import WarpOuterVecEnv
    root = Path(root)
    p = verify(root)
    cfgs = read(root / 'configs.json')
    if (arm, seed) not in [(r['arm'], r['seed']) for r in p['records']]:
        raise ValueError('Undeclared job')
    cfg = cfgs[arm]
    spec = ARMS[arm]
    out = root / f'{arm}-seed{seed}'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    inner = DDPG.load(root / 'inner_model.zip', device='cpu')
    inner.policy.set_training_mode(False)
    for parameter in inner.policy.parameters():
        parameter.requires_grad_(False)
    from stable_baselines3 import DDPG as _DDPG
    inner_digest = evaluation.model_digest(inner)
    control = 'iasa' if spec['outputs'] == 2 else 'direct'
    venv = WarpOuterVecEnv(cfg['plant'], cfg['inner_study'], _cuda_inner(inner), cfg['training_cases'][0],
                           n_envs=cfg['n_envs'], seed=seed, reward_shape='l1', failure=-5200.0,
                           cases=cfg['training_cases'], memory_divisor=cfg['reward'].get('memory_divisor', 0.5),
                           memory_cost=cfg['reward'].get('memory_cost', 0.5),
                           smooth_alpha=0.5 if control == 'iasa' else 1.0, backend='device', control=control)
    import time
    import resource
    started = time.perf_counter()
    manifest = dict(status='running', arm=arm, label=arm, seed=seed, control=control,
        protocol_sha256=sha(root / 'protocol.json'), inner_model_sha256=sha(root / 'inner_model.zip'))
    original.atomic(out / 'manifest.json', manifest)
    try:
        model = learner(venv, LABEL_ARCH[arm], seed, cfg)
        assert model.replay_buffer.size() == 0
        model.save(out / 'initial_model')
        manifest.update(initial_replay_size=0, initial_model_sha256=sha(out / 'initial_model.zip'),
            actor_parameters=sum(x.numel() for x in model.actor.parameters()))
        original.atomic(out / 'manifest.json', manifest)
        cb = Progress(out, root, cfg, inner, spec['contract'])
        model.learn(total_timesteps=cfg['total_timesteps'], callback=cb)
        model.save(out / 'last_model')
        model.save_replay_buffer(out / 'replay_buffer.pkl')
        if sha(root / 'inner_model.zip') != p['frozen_files']['inner_model.zip']:
            raise ValueError('Inner checkpoint changed')
        if evaluation.model_digest(inner) != inner_digest:
            raise ValueError('Inner parameters changed')
        manifest.update(status='completed', outer_actions=model.num_timesteps, physics_steps=cb.physics,
            gradient_updates=model._n_updates, selected=cb.best, failed_episodes=cb.failures,
            episodes=cb.episodes, final_cases=0, stop_reason='budget_ceiling')
    except BaseException as exc:
        manifest.update(status='failed', failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        venv.close()
        manifest['elapsed_s'] = time.perf_counter() - started
        manifest['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        original.atomic(out / 'status.json', dict(status=manifest['status'], elapsed_s=manifest['elapsed_s']))
        manifest['artifacts_sha256'] = {str(f.relative_to(out)): sha(f) for f in out.rglob('*') if f.is_file() and f.name != 'manifest.json'}
        original.atomic(out / 'manifest.json', manifest)


def _cuda_inner(inner_cpu):
    import tempfile
    tmp = tempfile.mkdtemp(prefix='iasa-inner')
    path = str(Path(tmp) / 'inner_model.zip')
    inner_cpu.save(path)
    from stable_baselines3 import DDPG as _DDPG
    cuda = _DDPG.load(path, device='cuda')
    cuda.policy.set_training_mode(False)
    for parameter in cuda.policy.parameters():
        parameter.requires_grad_(False)
    return cuda


def execute(output, workers=None):
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
        cfgs = read(output / 'configs.json')
        inner = DDPG.load(output / 'inner_model.zip', device='cpu')
        # PI baselines always run the frozen contract: PI never consumes the
        # 7/19-observation (it uses state_reference + its own loop), so its
        # traces are contract-independent. Only RL compare is patched per arm.
        for arm, spec in ARMS.items():
            evaluation.baselines(cfgs[arm]['plant'], cfgs[arm]['inner_study'], inner,
                                 cfgs[arm]['validation_cases'], output / cfgs[arm]['baseline_dir'])
        original.atomic(output / 'status.json', dict(status='running'))

        def job(r):
            started = now()
            try:
                with (output / 'logs' / f"{r['job']}.log").open('x') as log:
                    process = subprocess.run([sys.executable, '-m', 'tools.outer_iasa_v1', 'run', '--output', str(output),
                        '--arm', r['arm'], '--seed', str(r['seed'])], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
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
            or {(r['arm'], r['seed']) for r in queue} != {(r['arm'], r['seed']) for r in p['records']}
            or any(r['returncode'] for r in queue)):
        raise RuntimeError('Incomplete queue')
    rows = []
    for r in p['records']:
        out = output / r['job']
        m = read(out / 'manifest.json')
        if m['status'] != 'completed':
            raise RuntimeError('Incomplete job')
        for rel, digest in m['artifacts_sha256'].items():
            if sha(out / rel) != digest:
                raise ValueError('Changed training artifact')
        if sha(out / 'best_model.zip') != m['selected']['sha256']:
            raise ValueError('Changed selected model')
        rows.append(dict(**r, selected=m['selected'], outer_actions=m['outer_actions'],
                         failed_episodes=m['failed_episodes']))
    by_arm = {}
    for row in rows:
        by_arm.setdefault(row['arm'], []).append(row)
    scores = {a: [sum(x['selected']['passed_cases'] != x['selected']['case_count'] for x in rs), 0,
                  float(np.median([x['selected']['score'][2] for x in rs]))] for a, rs in by_arm.items()}
    result = dict(rows=rows, scores=scores, winner=min(scores, key=scores.get), final_qualification=False)
    save(output / 'selection.json', result)
    (output / 'selection.sha256').write_text(sha(output / 'selection.json') + '\n')
    lines = ['# Outer IASA development comparison', '', f"Validation-ranked arm: {result['winner']}. No final qualification.", '',
             '| Arm | Seed | Selected step | Cases passed | Speed RMSE rad/s |', '|---|---:|---:|---:|---:|']
    for row in rows:
        s = row['selected']
        lines.append(f"| {row['arm']} | {row['seed']} | {s['step']} | {s['passed_cases']}/{s['case_count']} | {s['score'][2]:.6f} |")
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['prepare', 'run', 'execute', 'select'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inner', type=Path, default=Path('results/bldc/outer-inner-freeze-v1'))
    ap.add_argument('--arm', choices=list(ARMS))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare(a.output, a.inner)
    elif a.command == 'run':
        run_arm(a.output, a.arm, a.seed)
    elif a.command == 'execute':
        execute(a.output, a.workers)
    else:
        select(a.output)


if __name__ == '__main__':
    main()
