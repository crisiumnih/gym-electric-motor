# Outer IASA development screen (Warp trains, contract GEM validates)

## Design (plan/plan.md sect.2, staged)

Two arms, one root (`outer-iasa-v1`), read post-hoc as repair-then-architecture:
- `direct-td3` (seeds 32/33): TD3 scalar 7-obs v2-memory L1/-5200. Repair-effect
  control: per-lane 40-case coverage, `TimeLimit.truncated` bootstrapping,
  8 gradient updates per transition (update ratio matched across backends).
- `iasa-td3` (seeds 34/35): TD3 two-output 19-obs v3 IASA (direct gain 1.5 A,
  increment 0.015 A/action, anti-windup 0.1, filter 0.5, lags 5/10/20 ms,
  tanh(err/0.05) small-error feature). Reward: L1 family without z-memory
  term (no z exists in v3); candidate reward is the next separate stage.
Shared: gamma .995, 2x128/2x256, 250k/env x8, 25k/env validation, frozen
inner, effort pricing x1, unsmoothed direct arm (alpha 1.0).

## Contract dispatch (plan sect.1 repair)

- Training (Warp device loop): `control` flag per arm; v3 state (bias,
  histories, 19-obs, 2-output actions) implemented device-side with parity
  tests against the shared reference (`rl/outer/controller.py`).
- Validation/selection: contract-dispatched GEM adapter
  (`tools/outer_env_contract.py`: v1 → frozen OuterEnv, v2 → FixV1Env,
  v3 → IASAEnv). PI baselines ALWAYS run the frozen adapter (PI never
  consumes the 7/19-obs; patching it breaks PI rollouts — tested).
- Configs, contract versions, and vroom/GEM source hashes recorded in protocol;
  contract hash belongs in checkpoint metadata (runner records protocol hash).

## Known gap (documented, not hidden)

The NumPy reference backend (`WarpOuterBackend`) has no IASA path yet; only
the device loop trains IASA. Device↔reference parity for IASA rests on the
shared-reference unit tests plus the independent GEM adapter as judge. Full
NumPy IASA lands before any fresh qualification claim.

## Gates

Validation-only selection; failures block; no final qualification. Improvement
read: `iasa-td3` vs `direct-td3` (matched: same algorithm, reward family,
budget, coverage) on violations then median RMSE, plus relay/lazy trace read.
6/6 both seeds on either arm triggers the NEXT stage (candidate reward), not
qualification.

```bash
.venv/bin/python -m pytest -q tests/test_outer_iasa_v1.py
.venv/bin/python -m tools.outer_iasa_v1 prepare --output results/bldc/outer-iasa-v1
.venv/bin/python -m tools.outer_iasa_v1 execute --output results/bldc/outer-iasa-v1
```
