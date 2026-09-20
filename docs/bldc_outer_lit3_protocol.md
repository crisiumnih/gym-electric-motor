# Lit-backed staircase protocols (Warp training, GEM validation)

Three new lineages, one variable per step (launched in parallel, read as
staircase post-hoc). Shared: v2 memory contract, gamma .995, 2x128,
250k/env x8, 25k/env validation, frozen inner, effort pricing x1.

- `outer-td3-v1` (seeds 26/27): TD3 learner; L1, -5200. Algorithm thesis
  (twin critics vs DDPG overestimation; cited SSE <0.5%).
- `outer-qeff-v1` (seeds 28/29): TD3 + quadratic speed cost qw=100
  (calibrated: equals L1 cost at en=0.04; smooth gradient at zero).
  Reward thesis (L1 kink implicated in relay).
- `outer-smooth-v1` (seeds 30/31): TD3 + qeff + plant-facing command
  smoothing alpha=0.5 (delta penalty stays on raw output). Chatter thesis.

Pre-declared read per screen: validation selection only; improvement =
lower median violations AND lower median RMSE than the previous staircase
step (td3 vs fix-warp partial/GEM history qualitatively, qeff vs td3,
smooth vs qeff). Any 6/6 both seeds triggers frozen fresh qualification
and only then a `main` architecture PR.

```bash
.venv/bin/python -m pytest -q tests/test_outer_lit3.py
.venv/bin/python -m tools.outer_lit3 prepare --screen td3 --output results/bldc/outer-td3-v1
.venv/bin/python -m tools.outer_lit3 prepare --screen qeff --output results/bldc/outer-qeff-v1
.venv/bin/python -m tools.outer_lit3 prepare --screen smooth --output results/bldc/outer-smooth-v1
```
