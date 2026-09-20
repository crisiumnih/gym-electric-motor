# Outer warp ceiling protocol (new lineage)

Continues the g0995 winner values (gamma .995, L1 tracking, -5200 failure,
frozen inner seed 9/500k) at 500k per-env budget and 3 widths (128x2, 256x3,
512x3), fresh seeds 20/21, validation every 50k per-env.

## Split design

- Training rollouts: `WarpOuterBackend` (CUDA physics + batched torch inner),
  8 parallel envs cycling the 40 GEM training cases (same distribution as the
  CPU lineage; one shared case per episode across envs).
- Budget/validation/learning-starts are per-env (GEM semantics); SB3 global
  counts scale x8 (4M total steps for 500k/env). Manifests record `n_envs`.
- Validation/selection/baselines: frozen GEM stack, unchanged.
- New lineage `outer-warp-ceiling-v1`: no comparison claims vs CPU runs.

## Saturation reading

Validation curves decide the ceiling: selection every 50k, plateau analysis
post-hoc. Width ranking by failed seeds, violations, median RMSE.

## Reproduce

```bash
.venv/bin/python -m tools.outer_warp_ceiling prepare --output results/bldc/outer-warp-ceiling-v1
.venv/bin/python -m tools.outer_warp_ceiling execute --output results/bldc/outer-warp-ceiling-v1
```
