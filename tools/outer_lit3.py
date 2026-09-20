"""Lit-backed screen staircase on Warp training + GEM validation.

Three new lineages, one variable each step (staircase; launched in parallel,
read as staircase post-hoc):
- outer-td3-v1: TD3 learner; v2 memory, L1, -5200 (algorithm thesis).
- outer-qeff-v1: TD3 + quadratic speed cost qw=100, same effort pricing
  (reward thesis; smooth gradient at zero vs L1 kink).
- outer-smooth-v1: TD3 + qeff + command smoothing alpha=0.5 plant-facing
  (chatter thesis; delta penalty stays on raw output).
Shared: gamma .995, 2x128, 250k/env x8, 25k/env validation, frozen inner,
fresh seeds. Frozen GEM sources imported, never edited.
"""
import argparse
from pathlib import Path

from benchmarks.bldc.run import sha, ROOT
from benchmarks.bldc.current_rl.repeat_study import read, save
from tools.outer_rl import study as original
from tools import outer_warp_ceiling as ceiling

LABEL = '128x2'
BUDGET = 250000
VALID_EVERY = 25000
N_ENVS = 8
BASE_REWARD = {'memory_divisor': 0.05, 'memory_cost': 0.005}
BASE_LEARNER = {}

SCREENS = {
    'td3': dict(protocol='outer-td3-v1', seeds=(26, 27),
                reward_extra=dict(BASE_REWARD),
                learner_extra=dict(algorithm='td3'),
                note='TD3 learner; v2 memory, L1, -5200'),
    'qeff': dict(protocol='outer-qeff-v1', seeds=(28, 29),
                 reward_extra=dict(BASE_REWARD, shape='qeff', quad_weight=100.0),
                 learner_extra=dict(algorithm='td3'),
                 note='TD3 + quadratic speed cost; same effort pricing'),
    'smooth': dict(protocol='outer-smooth-v1', seeds=(30, 31),
                   reward_extra=dict(BASE_REWARD, shape='qeff', quad_weight=100.0, smooth_alpha=0.5),
                   learner_extra=dict(algorithm='td3'),
                   note='TD3 + qeff + plant-facing command smoothing 0.5'),
}


def prepare(output, screen, inner, budget=BUDGET, seeds=None, validation_every=VALID_EVERY, max_workers=2, learner_extra=None):
    sc = SCREENS[screen]
    output = Path(output).resolve()
    merged_learner = dict(sc['learner_extra'])
    merged_learner.update(learner_extra or {})
    p = ceiling.prepare(output, inner, budget=budget, seeds=list(seeds or sc['seeds']), widths=(LABEL,),
                        validation_every=validation_every, max_workers=max_workers,
                        reward_extra=sc['reward_extra'], protocol_name=sc['protocol'], n_envs=N_ENVS,
                        learner_extra=merged_learner,
                        interpretation_note=f'; lit3 {screen}: {sc["note"]}')
    cfg = read(output / 'config.json')
    cfg['contract'].update(version='bldc-outer-speed-v2', memory_time_s=0.05)
    save(output / 'config.json', cfg)
    proto = read(output / 'protocol.json')
    proto['config'] = cfg
    import hashlib
    import json
    proto['config_sha256'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    proto['frozen_files']['config.json'] = sha(output / 'config.json')
    proto['source_sha256'][str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    proto['screen'] = screen
    save(output / 'protocol.json', proto)
    (output / 'protocol.sha256').write_text(sha(output / 'protocol.json') + '\n')
    return read(output / 'protocol.json')


def verify(output, screen):
    p = ceiling.verify(output, SCREENS[screen]['protocol'])
    if p.get('screen') != screen:
        raise ValueError('Wrong lit3 screen')
    c = read(Path(output) / 'config.json')
    if c['contract'].get('version') != 'bldc-outer-speed-v2':
        raise ValueError('Wrong v2 contract')
    return p


def run(output, screen, label, seed):
    ceiling.run_arm(Path(output), label, seed, SCREENS[screen]['protocol'])


def execute(output, screen, workers=None):
    ceiling.execute(Path(output).resolve(), workers, SCREENS[screen]['protocol'])


def select(output, screen):
    return ceiling.select(Path(output), SCREENS[screen]['protocol'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['prepare', 'run', 'execute', 'select'])
    ap.add_argument('--screen', choices=list(SCREENS), required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inner', type=Path, default=Path('results/bldc/outer-inner-freeze-v1'))
    ap.add_argument('--label', default=LABEL)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare(a.output, a.screen, a.inner)
    elif a.command == 'run':
        run(a.output, a.screen, a.label, a.seed)
    elif a.command == 'execute':
        execute(a.output, a.screen, a.workers)
    else:
        select(a.output, a.screen)


if __name__ == '__main__':
    main()
