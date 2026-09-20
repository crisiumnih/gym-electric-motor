# Outer warp ceiling protocol (new lineage)

Continues the g0995 winner values (gamma .995, L1 tracking, -5200 failure,
frozen inner seed 9/500k) at 500k budget and 3 widths (128x2, 256x3, 512x3),
fresh seeds 20/21, validation every 50k.

## Split design

- Training rollouts: `WarpOuterBackend` (CUDA physics + batched torch inner).
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
