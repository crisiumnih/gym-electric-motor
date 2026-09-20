"""Outer memory-gain fix trained on Warp, validated on GEM (new lineage).

Same one-variable screen as outer-fix-v1 (v2 contract: 10x memory gain,
rescaled cost) but Warp-trained per the standing directive (GEM never trains).
Comparison target: horizon-v2 g0995 (GEM-trained) and the killed GEM fix-v1
partial run — backend is an acknowledged second variable, recorded here;
the judge (frozen GEM validation) is identical, so gate outcomes still decide.

Lineage: outer-fix-warp-v1. Frozen GEM sources imported, never edited.
"""
import argparse
from pathlib import Path

from benchmarks.bldc.run import sha, ROOT
from benchmarks.bldc.current_rl.repeat_study import read, save
from tools.outer_rl import study as original
from tools import outer_warp_ceiling as ceiling

PROTOCOL = 'outer-fix-warp-v1'
LABEL = '128x2'
SEEDS = (24, 25)
BUDGET = 250000
VALID_EVERY = 25000
N_ENVS = 8
MEMORY_DIVISOR = 0.05
MEMORY_COST = 0.005
CONTRACT = 'bldc-outer-speed-v2'


def prepare(output, inner, budget=BUDGET, seeds=SEEDS, validation_every=VALID_EVERY, max_workers=2, learner_extra=None):
    output = Path(output).resolve()
    p = ceiling.prepare(output, inner, budget=budget, seeds=list(seeds), widths=(LABEL,),
                        validation_every=validation_every, max_workers=max_workers,
                        reward_extra={'memory_divisor': MEMORY_DIVISOR, 'memory_cost': MEMORY_COST},
                        protocol_name=PROTOCOL, n_envs=N_ENVS, learner_extra=learner_extra,
                        interpretation_note='; fix-warp: v2 memory contract on Warp training')
    cfg = read(output / 'config.json')
    cfg['contract'].update(version=CONTRACT, memory_time_s=MEMORY_DIVISOR)
    cfg['interpretation'] += '; v2 contract (10x gain, rescaled cost)'
    save(output / 'config.json', cfg)
    for key in ('protocol.json',):
        proto = read(output / key)
        proto['config'] = cfg
        import hashlib
        import json
        proto['config_sha256'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        proto['frozen_files']['config.json'] = sha(output / 'config.json')
        proto['source_sha256'][str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
        proto['contract'] = dict(version=CONTRACT, memory_divisor=MEMORY_DIVISOR, memory_cost=MEMORY_COST)
        save(output / key, proto)
    (output / 'protocol.sha256').write_text(sha(output / 'protocol.json') + '\n')
    return read(output / 'protocol.json')


def verify(output):
    p = ceiling.verify(output, PROTOCOL)
    c = read(Path(output) / 'config.json')
    if c['contract'].get('version') != CONTRACT or c['contract'].get('memory_time_s') != MEMORY_DIVISOR:
        raise ValueError('Wrong v2 contract')
    if c['reward'].get('memory_divisor') != MEMORY_DIVISOR or c['reward'].get('memory_cost') != MEMORY_COST:
        raise ValueError('Wrong memory dispatch')
    if c['reward'].get('shape') != 'l1' or c['reward'].get('failure') != -5200.0:
        raise ValueError('Wrong shared reward')
    return p


def run(output, label, seed):
    ceiling.run_arm(Path(output), label, seed, PROTOCOL)


def execute(output, workers=None):
    ceiling.execute(Path(output).resolve(), workers, PROTOCOL)


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
