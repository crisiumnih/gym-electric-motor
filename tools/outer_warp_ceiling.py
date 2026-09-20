"""Outer ceiling study on Warp training + GEM validation (new lineage).

Continues the g0995 winner config (gamma .995, L1 tracking, -5200 failure,
frozen inner) at longer budget and wider capacity screen to find saturation.
Training rolls out on WarpOuterBackend (CUDA); ALL validation/selection uses
the frozen GEM stack. No comparison claims against CPU-lineage runs.

Lineage: outer-warp-ceiling-v1. This file + warp_spike backend sources are
hashed into the protocol; frozen GEM sources are imported, never edited.
"""
import argparse
import copy
import hashlib
import json
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from benchmarks.bldc.run import ROOT, sha
from benchmarks.bldc.current_rl.repeat_study import read, save, now
from tools.outer_rl import study as original, evaluation
from tools import outer_horizon_study as horizon

WIDTHS = {'128x2': [128, 128], '256x3': [256] * 3, '512x3': [512] * 3}
PROTOCOL = 'outer-warp-ceiling-v1'
GAMMA = 0.995
FAILURE = -5200.0
WARP_SOURCES = ['tools/outer_warp_ceiling.py']
WARP_BACKEND_FILES = ['backend.py', 'fullstep.py', 'vecenv.py', 'kernels.py', 'kernels64.py']


def warp_backend_root():
    cand = ROOT / 'third_party' / 'gym-electric-motor'
    base = Path('/home/sra/prajwal/vroom/warp_backend')
    return base if base.is_dir() else None


def warp_backend_digest():
    base = warp_backend_root()
    h = hashlib.sha256()
    for name in sorted(WARP_BACKEND_FILES):
        h.update(sha(base / name).encode())
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    save(temp, value)
    temp.replace(path)


def score(rows):
    return [sum(not r['rl']['completed'] for r in rows),
            sum(len(r['violations']) for r in rows),
            float(np.mean([r['rl'].get('rmse_rad_s', float('inf')) for r in rows]))]


class Progress(BaseCallback):
    def __init__(self, output, root, config, inner):
        super().__init__()
        self.out = output
        self.root = root
        self.c = config
        self.inner = inner
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
            with horizon.environment('g0995'):
                rows = evaluation.compare(self.model, self.c['plant'], self.c['inner_study'], self.inner,
                    self.c['validation_cases'], self.out / 'validation' / str(self.num_timesteps),
                    baseline_directory=self.root / 'baselines')
            value = score(rows)
            if not np.isfinite(value[-1]):
                raise RuntimeError('Nonfinite or missing validation RMSE')
            if self.best is None or value < self.best['score']:
                self.model.save(self.out / 'best_model')
                self.best = dict(step=self.num_timesteps, score=value,
                    sha256=sha(self.out / 'best_model.zip'),
                    passed_cases=sum(r['passed'] for r in rows), case_count=len(rows))
                atomic(self.out / 'selection.json', self.best)
        if self.num_timesteps % 1000 == 0:
            atomic(self.out / 'status.json', dict(status='running', outer_actions=self.num_timesteps,
                physics_steps=self.physics, updates=self.model._n_updates, episodes=self.episodes,
                failed_episodes=self.failures, selected=self.best,
                elapsed_s=time.perf_counter() - self.started))
        return True


def learner(env, widths, seed, cfg):
    c = cfg['learner']
    return DDPG('MlpPolicy', env, seed=seed, device=c.get('device', 'cuda'), learning_rate=c['learning_rate'],
        gamma=c['gamma'], tau=c['tau'], buffer_size=c['buffer_size'], learning_starts=c['learning_starts'],
        batch_size=c['batch_size'], train_freq=c['train_freq'], gradient_steps=c['gradient_steps'],
        action_noise=NormalActionNoise(np.zeros(1), np.ones(1) * c['noise_sigma']),
        policy_kwargs=dict(net_arch=dict(pi=widths, qf=c['critic_widths']), activation_fn=torch.nn.ReLU))


def run_arm(root, label, seed):
    sys.path.insert(0, str(warp_backend_root().parent))
    from warp_backend.vecenv import WarpOuterVecEnv
    root = Path(root)
    p = verify(root)
    cfg = read(root / 'config.json')
    if (label, seed) not in [(r['label'], r['seed']) for r in p['records']]:
        raise ValueError('Undeclared job')
    out = root / f'{label}-seed{seed}'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    inner = DDPG.load(root / 'inner_model.zip', device='cpu')
    inner.policy.set_training_mode(False)
    for parameter in inner.policy.parameters():
        parameter.requires_grad_(False)
    inner_digest = evaluation.model_digest(inner)
    case = cfg['training_cases']
    venv = WarpOuterVecEnv(cfg['plant'], cfg['inner_study'], inner_cuda_holder(inner), case[0], n_envs=cfg['n_envs'],
                           seed=seed, reward_shape=cfg['reward'].get('shape', 'l2'),
                           failure=cfg['reward'].get('failure', -1040.0),
                           effort_scale=cfg['reward'].get('effort_scale', 1.0), cases=case)
    started = time.perf_counter()
    manifest = dict(status='running', label=label, seed=seed, protocol_sha256=sha(root / 'protocol.json'),
        inner_model_sha256=sha(root / 'inner_model.zip'))
    atomic(out / 'manifest.json', manifest)
    try:
        model = learner(venv, WIDTHS[label], seed, cfg)
        assert model.replay_buffer.size() == 0
        model.save(out / 'initial_model')
        manifest.update(initial_replay_size=0, initial_model_sha256=sha(out / 'initial_model.zip'),
            actor_parameters=sum(x.numel() for x in model.actor.parameters()))
        atomic(out / 'manifest.json', manifest)
        cb = Progress(out, root, cfg, inner)
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
        atomic(out / 'status.json', dict(status=manifest['status'], elapsed_s=manifest['elapsed_s']))
        manifest['artifacts_sha256'] = {str(f.relative_to(out)): sha(f) for f in out.rglob('*')
                                        if f.is_file() and f.name != 'manifest.json'}
        atomic(out / 'manifest.json', manifest)


def inner_cuda_holder(inner_cpu):
    import tempfile
    tmp = tempfile.mkdtemp(prefix='warp-inner')
    path = str(Path(tmp) / 'inner_model.zip')
    inner_cpu.save(path)
    cuda = DDPG.load(path, device='cuda')
    cuda.policy.set_training_mode(False)
    for parameter in cuda.policy.parameters():
        parameter.requires_grad_(False)
    return cuda


def prepare(output, inner, budget=500000, seeds=(20, 21), widths=('128x2', '256x3', '512x3'),
            validation_every=50000, max_workers=4, reward_extra=None, protocol_name=None,
            interpretation_note='', n_envs=8, learner_extra=None):
    """Budget/validation_every are PER-ENV outer actions (GEM semantics); SB3
    global counts are scaled by n_envs. Training cycles the 40 GEM cases."""
    import shutil
    output, inner = Path(output).resolve(), Path(inner).resolve()
    widths = list(widths)
    if output.exists():
        raise FileExistsError(output)
    if any(w not in WIDTHS for w in widths):
        raise ValueError('Invalid widths')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Invalid seeds')
    if n_envs <= 0:
        raise ValueError('Invalid n_envs')
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
    learner_cfg = dict(gamma=GAMMA, learning_rate=1e-4, batch_size=128, buffer_size=300000, learning_starts=5000 * n_envs,
                     tau=.005, train_freq=1, gradient_steps=1, noise_sigma=.05, critic_widths=[256, 256],
                     device='cuda')
    learner_cfg.update(learner_extra or {})
    cfg = dict(protocol='bldc-outer-warp-ceiling-v1', plant=saved['plant'], inner_study=saved['study'],
        training_cases=training_cases(), validation_cases=validation_cases(), test_cases=[],
        learner=learner_cfg,
        contract=dict(version='bldc-outer-speed-v1', outer_dt_s=.001, inner_dt_s=.0001, hold_steps=10,
                      speed_scale_rad_s=25., current_scale_a=4., reference_limit_a=1.5, memory_time_s=.5,
                      phase_limit_a=4.,
                      features=['speed', 'reference', 'error', 'id', 'iq', 'previous_own_command', 'bounded_error_memory']),
        reward=dict(shape='l1', failure=FAILURE, **(reward_extra or {}),
                    interpretation='L1 absolute tracking + shared failure penalty, g0995 lineage'),
        validation_every=validation_every * n_envs, total_timesteps=budget * n_envs,
        n_envs=n_envs, per_env_timesteps=budget,
        interpretation=('Warp-training/GEM-validation ceiling screen; gamma .995/L1/-5200 from g0995 winner; '
                        'longer budget + wider capacity; 8 parallel envs cycling the 40 GEM training cases; '
                        'budget/validation/starts are per-env, SB3 global counts scaled x8; '
                        'new lineage, no CPU-run comparison' + interpretation_note))
    save(output / 'config.json', cfg)
    import importlib.metadata
    from benchmarks.bldc.run import provenance
    p = provenance(cfg)
    p['packages'].update({n: importlib.metadata.version(n) for n in ['torch', 'stable-baselines3']})
    for f in sorted((ROOT / 'tools/outer_rl').glob('*.py')):
        p['source_sha256'][str(f.relative_to(ROOT))] = sha(f)
    for rel in WARP_SOURCES + ['tools/outer_horizon_study.py', 'tools/outer_reward_study.py']:
        cand = ROOT / rel
        p['source_sha256']['warp:' + rel] = sha(cand)
    p['source_sha256']['warp:backend'] = warp_backend_digest()
    p['warp_backend_root'] = str(warp_backend_root())
    p.update(protocol=protocol_name or PROTOCOL, seeds=list(seeds), widths=widths, max_workers=max_workers,
        budget=budget, n_envs=n_envs, max_outer_actions=len(widths) * len(seeds) * budget,
        max_training_physics_steps=10 * len(widths) * len(seeds) * budget,
        device=dict(learner='cuda', inner='cuda', physics='warp-cuda', validation='gem-cpu'),
        frozen_files={n: sha(output / n) for n in ['config.json', 'inner_model.zip', 'inner_config.json', 'inner_freeze.json']},
        records=[dict(label=w, seed=s, arm=f'{w}-seed{s}') for w in widths for s in seeds],
        selection='Failed seeds, incomplete cases, gate violations, median speed RMSE; saturation read from validation curves',
        initialization='SB3 default random actor/critic; empty replay; no PI labels or warm start',
        lineage='New Warp-training lineage continuing g0995 config values; no CPU-lineage comparison claims',
        final_evaluation=False)
    save(output / 'protocol.json', p)
    (output / 'protocol.sha256').write_text(sha(output / 'protocol.json') + '\n')
    atomic(output / 'status.json', dict(status='prepared'))
    return p


def verify(output):
    output = Path(output)
    p = read(output / 'protocol.json')
    if sha(output / 'protocol.json') != (output / 'protocol.sha256').read_text().strip():
        raise ValueError('Changed ceiling protocol')
    if p.get('protocol') != PROTOCOL:
        raise ValueError('Not a ceiling protocol')
    import importlib.metadata
    for package, version in p['packages'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'Changed package {package}')
    for name, digest in p['frozen_files'].items():
        if sha(output / name) != digest:
            raise ValueError(f'Changed frozen input {name}')
    for rel, digest in p['source_sha256'].items():
        if rel == 'warp:backend':
            if warp_backend_digest() != digest:
                raise ValueError(f'Changed source {rel}')
            continue
        base = rel[5:] if rel.startswith('warp:') else rel
        cand = ROOT / base
        if sha(cand) != digest:
            raise ValueError(f'Changed source {rel}')
    return p


def execute(output, workers=None):
    output = Path(output).resolve()
    p = verify(output)
    workers = workers or p.get('max_workers', 4)
    with (output / 'queue_started.json').open('x') as f:
        json.dump(dict(started_utc=now()), f)
    (output / 'logs').mkdir(exist_ok=False)
    atomic(output / 'status.json', dict(status='preparing_baselines'))
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cfg = read(output / 'config.json')
        inner = DDPG.load(output / 'inner_model.zip', device='cpu')
        with horizon.environment('g0995'):
            evaluation.baselines(cfg['plant'], cfg['inner_study'], inner,
                                 cfg['validation_cases'], output / 'baselines')
        atomic(output / 'status.json', dict(status='running'))

        def job(r):
            started = now()
            try:
                with (output / 'logs' / f"{r['arm']}.log").open('x') as log:
                    process = subprocess.run(
                        [sys.executable, '-m', 'tools.outer_warp_ceiling', 'run', '--output', str(output),
                         '--label', r['label'], '--seed', str(r['seed'])],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                return dict(**r, returncode=process.returncode, started_utc=started, completed_utc=now())
            except Exception as exc:
                return dict(**r, returncode=-1, failure=f'{type(exc).__name__}: {exc}')
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(job, r) for r in p['records']]
            for future in as_completed(futures):
                results.append(future.result())
                atomic(output / 'queue_status.json', dict(completed_jobs=len(results),
                       total_jobs=len(p['records']), results=results))
                print(json.dumps(results[-1]), flush=True)
        save(output / 'run_summary.json', results)
        if any(r['returncode'] for r in results):
            raise RuntimeError('Failed job; no aggregate selection')
        select(output)
        atomic(output / 'status.json', dict(status='completed', completed_utc=now()))
    except BaseException as exc:
        atomic(output / 'status.json', dict(status='failed',
               failure=f'{type(exc).__name__}: {exc}', failed_utc=now()))
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
    rows = []
    for r in p['records']:
        out = output / r['arm']
        m = read(out / 'manifest.json')
        if m['status'] != 'completed':
            raise RuntimeError('Incomplete job')
        for rel, digest in m['artifacts_sha256'].items():
            if sha(out / rel) != digest:
                raise ValueError('Changed training artifact')
        if sha(out / 'best_model.zip') != m['selected']['sha256']:
            raise ValueError('Changed selected model')
        rows.append(dict(label=r['label'], seed=r['seed'], arm=r['arm'], selected=m['selected'],
                         outer_actions=m['outer_actions'], failed_episodes=m['failed_episodes']))
    by_label = {}
    for row in rows:
        by_label.setdefault(row['label'], []).append(row)
    scores = {w: [min(1, 6 - max(r['selected']['passed_cases'] for r in rs)), 0,
                  float(np.median([r['selected']['score'][2] for r in rs]))]
              for w, rs in by_label.items()}
    result = dict(rows=rows, scores=scores, winner=min(scores, key=scores.get),
                  final_qualification=False,
                  all_winner_seeds_pass_validation=all(
                      r['selected']['passed_cases'] == r['selected']['case_count']
                      for r in rows if r['label'] == min(scores, key=scores.get)))
    save(output / 'selection.json', result)
    (output / 'selection.sha256').write_text(sha(output / 'selection.json') + '\n')
    lines = ['# Outer warp ceiling development comparison', '',
             f"Validation-ranked width: {result['winner']}. No final qualification.", '',
             '| Width | Seed | Selected step | Cases passed | Speed RMSE rad/s |', '|---|---:|---:|---:|---:|']
    for row in rows:
        s = row['selected']
        lines.append(f"| {row['label']} | {row['seed']} | {s['step']} | {s['passed_cases']}/{s['case_count']} | {s['score'][2]:.6f} |")
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['prepare', 'run', 'execute', 'select'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inner', type=Path, default=Path('results/bldc/outer-inner-freeze-v1'))
    ap.add_argument('--label')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare(a.output, a.inner)
    elif a.command == 'run':
        run_arm(a.output, a.label, a.seed)
    elif a.command == 'execute':
        execute(a.output, a.workers)
    else:
        select(a.output)


if __name__ == '__main__':
    main()
