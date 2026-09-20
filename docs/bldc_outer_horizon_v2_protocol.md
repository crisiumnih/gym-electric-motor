# Outer discount-horizon v2 (CUDA rerun) protocol

Reruns the frozen v1 comparison (`docs/bldc_outer_horizon_protocol.md`: gamma
.995 vs .999, L1 tracking, shared −5200 failure penalty, 2x128 actor / 2x256
critic, frozen inner seed 9/500k) with one controlled change: compute placement.

## Controlled difference from v1

- Outer DDPG gradient updates: `device='cuda'` (RTX 3060, torch 2.13.0+cu130).
- Frozen inner rollout policy: `device='cpu'` (10 kHz tiny-forward loop; per-step
  transfer would dominate).
- Fresh seeds 18/19 (16/17 consumed by stopped v1). Budgets, reward, timing,
  encoder, scenarios, gates unchanged. No CUDA↔CPU bit-identity claimed.

## Implementation

- New adapter `tools/outer_horizon_v2.py` (v1 files untouched). `cuda_learner`
  mirrors `original.learner`; `run` patches it in alongside the horizon
  environment patch. `verify` additionally asserts matched `learner.device`.
- `prepare` builds children straight from `outer-inner-freeze-v1` via
  `original.prepare` (the v1 reward-parent chain is absent on this host) and,
  with `strict=True`, asserts each fresh child config equals the matching
  stopped-v1 child config on disk except `learner.device`/`interpretation`.
- Root: `results/bldc/outer-horizon-v2`, protocol `outer-discount-horizon-v2`,
  fresh seeds 18/19, `max_workers=4` (seeded subprocesses; scheduling does not
  affect numerics).

## Gates

Same as v1: validation-only selection every 25k, 0/6 development cases expected
until proven otherwise; no final qualification; failures block selection.

## Reproduce

```bash
.venv/bin/python -m pytest -q tests/test_outer_horizon_v2.py
.venv/bin/python -m tools.outer_horizon_v2 prepare --output results/bldc/outer-horizon-v2
.venv/bin/python -m tools.outer_horizon_v2 execute --output results/bldc/outer-horizon-v2
```
