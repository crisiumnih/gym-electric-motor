# Outer memory-gain fix (v1) protocol

## Mechanism (measured, not hypothesized)

On horizon-v2 winner traces (g0999-s19 @125k, reversal, persistent 0.26 rad/s
bias over 2000 outer actions): reconstructed v1 memory satisfies |z| <= 0.018,
cost <= 2e-5/step. The accumulator gains 0.002*en per action, so errors at the
observed scale (en ~ 0.01) never move it. The actor has no integral signal.
Correction economics at that error: holding costs ~0.083/step (speed + L1
delta); a full corrective swing costs 0.025/step delta over the hold plus
torque-transient penalties plus continued speed cost through the transient.
Same-actor command audit: rail-to-rail relay on negatives (up to 823 flips),
flat-lazy on positives (tail std 0.001). One root (no integral authority), two
symptoms. More epochs/width cannot fix a missing channel.

## Single variable

v2 contract `bldc-outer-speed-v2`: memory divisor 0.5 -> 0.05 (10x gain),
memory cost weight 0.5 -> 0.005. For identical error history pre-clip,
z_new = 10*z_old and 0.005*100*z_old^2 = 0.5*z_old^2: reward cost identical,
observation 10x. Proven by `test_gain_gives_10x_signal_at_identical_memory_cost`.
Everything else (gamma .995, L1, -5200, 2x128, 250k, 25k validation, CUDA
learner/CPU inner on GEM) matches horizon-v2 g0995. Fresh seeds 24/25.

## Design notes

- Trains on GEM (not Warp): the comparison target is GEM-trained v2-g0995, so
  the backend must stay fixed. Warp enters later screens.
- Baselines/validation use the v2 env via the same patch pattern as the
  horizon adapter; PI traces are unaffected (PI never consumes z), so
  physics metrics stay cross-study comparable.
- Pre-declared read: improvement = lower median violations AND lower median
  RMSE than v2-g0995 (0.711), plus relay/lazy trace read. 6/6 both seeds
  triggers frozen fresh qualification and only then a `main` architecture PR.

## Reproduce

```bash
.venv/bin/python -m pytest -q tests/test_outer_fix_v1.py
.venv/bin/python -m tools.outer_fix_v1 prepare --output results/bldc/outer-fix-v1
.venv/bin/python -m tools.outer_fix_v1 execute --output results/bldc/outer-fix-v1
```
