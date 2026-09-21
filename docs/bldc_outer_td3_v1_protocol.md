# Outer TD3 learner-swap (v1) protocol

## Single variable

Everything matches horizon-v2 g0995 (gamma .995, L1 tracking, -5200 failure,
v1 contract + v1 memory, 2x128/2x256, frozen inner, Warp 8-env training, GEM
validation) except the learner: SB3 TD3 instead of DDPG. TD3-specific hypers
are SB3 defaults, pre-declared: policy_delay=2, target_policy_noise=0.2,
target_noise_clip=0.5. Fresh seeds 26/27, 250k/env, validation every 25k/env.

## Why

Literature (TD3 vs DDPG motor-control table: overshoot -70%, SSE <0.5%,
stable reward) and our traces (DDPG chatter/stall, overestimation-prone
single critic) both indict vanilla DDPG. Twin critics + delayed actors +
target smoothing directly target the relay side; stable values help the lazy
side converge instead of collapsing.

## Read

Pre-declared: improvement = lower median violations AND lower median RMSE
than v2-g0995 (0.711) with relay/lazy trace audit. 6/6 both seeds triggers
frozen fresh qualification, and only then a `main` architecture PR.

## Reproduce

```bash
.venv/bin/python -m pytest -q tests/test_outer_td3_v1.py
.venv/bin/python -m tools.outer_td3_v1 prepare --output results/bldc/outer-td3-v1
.venv/bin/python -m tools.outer_td3_v1 execute --output results/bldc/outer-td3-v1
```
