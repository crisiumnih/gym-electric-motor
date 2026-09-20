# BLDC Model Verification

**Scope:** evidence that the BLDC model in `gym-electric-motor` implements the equations of
`docs/bldc_model.md` correctly. All figures in `figures/bldc/`, all numbers below are
reproducible with the commands at the end.

**Status:** v2 — all checks pass; one modeling gap found and fixed during verification
(common-mode current, decision D8 in `docs/bldc_model.md`). The derived dq observations now use
the electrical angle at the end of each integration step, and the verification figures use
non-overlapping layouts and edge-safe ripple envelopes.

---

## 1. Verification matrix

| Check | Test / artifact | Result |
|---|---|---|
| Back-EMF shape: plateaus ±1 over exactly 120° electrical, linear 60° transitions, zero crossings at 0°/180° | `test_BrushlessDCMotor_backemf_shape` | pass |
| Phase offsets exactly 120° electrical | same | pass |
| dq alignment: cycle mean (e_d, e_q) = (0, +1.216) | same | pass (|e_d| < 1e-9) |
| Back-EMF scaling e = k_e · ω · f | `test_BrushlessDCMotor_backemf` | pass |
| ODE steady states I = (U − E)/R, dε/dt = p·ω | `test_BrushlessDCMotor_el_ode` | pass |
| Star connection: Σi ≡ 0 (floating neutral) | `test_BrushlessDCMotor_star_connection`, `test_bldc_env_star_connection` | pass |
| Torque law T = k_e Σ f_j i_j and power balance T·ω = Σ e_j i_j | `test_BrushlessDCMotor_torque` + `fig_power_balance.png` | pass (max rel. err 1.8e-13 over 20,000 points) |
| Torque limit 2·k_e·i_lim | `test_BrushlessDCMotor_torque_limit` | pass |
| Limits / nominal value derivation | `test_BrushlessDCMotor_limits` | pass |
| Smoothing preserves plateaus, half-wave symmetry, bounded deviation | `test_BrushlessDCMotor_smoothing` | pass |
| Env: state vector identical to PMSM env, dq action (2-dim), reference omega, tau | `test_bldc_env_make` | pass |
| Env: abc action (3-dim) | `test_bldc_env_abc_action_space` | pass |
| Env: q-voltage spins the motor, d-voltage does not | `test_bldc_env_q_voltage_spins_up` | pass |
| Env: derived dq observations match abc values at returned epsilon | `test_bldc_env_dq_observations_use_returned_epsilon` | pass |
| Closed-loop speed control reaches the reference | `fig_closedloop_speed.png` + `examples/bldc/cont_sc_bldc_example.py` | pass (below) |
| Full regression suite | `pytest tests/` | 1241 passed, 52 warnings |

## 2. Figures (thesis-ready)

**Figure 1 — Trapezoidal back-EMF shape functions.** `fig_bemf_shape.png`. The phase-a shape
function of Eq. (10): flat ±1 over 2π/3, linear transitions over π/3, zero crossings at 0 and π
electrical, phases displaced by 2π/3. Right panel: the optional smoothing variant
(decision D3) preserves the plateaus exactly.

**Figure 2 — dq alignment of the back-EMF.** `fig_bemf_dq_alignment.png`. Left: cycle mean of
(e_d, e_q) of the shape function vs. the angle-origin shift φ; the choice φ = 210° places the
fundamental purely on the q-axis (e_d = 0, e_q = 1.216). Right: instantaneous e_d(ε), e_q(ε) at
φ = 210° — constant means with the 6th-harmonic ripple inherent to the trapezoid.

**Figure 3 — Open-loop spin-up.** `fig_openloop_spinup.png`. Constant q-axis voltage
u_q = 1.1 V; the speed rises toward the no-load ideal u_q/K_e = 11.5 rad/s (reduced by the
polynomial load); the translucent band is the ripple envelope (speed ripple from the 6th
harmonic, see below).

**Figure 4 — Closed-loop speed control.** `fig_closedloop_speed.png`. Clamped PI on u_q,
ω* = 10 rad/s, J = 3e-3 kg·m² (rotor + propeller-class load, see §3). The controller settles at
10.00 rad/s (steady-state error < 0.01 rad/s), speed ripple 0.06 rad/s p-p. Phase currents and
torque shown in a 0.2 s steady-state window: 6·f_e ≈ 200 Hz ripple at f_e = 33 Hz.

**Figure 5 — Torque ripple.** `fig_torque_ripple.png`. Left: intrinsic ripple of the machine
(unit sinusoidal q-current, torque from Eq. (12)): 14.7% of the mean torque — the idealized
lower bound. Right: simulated steady-state torque of the voltage-driven drive: p-p 0.25 N·m on
a load mean of 0.11 N·m. The difference is physical: the trapezoidal back-EMF injects 5th/7th
current harmonics (barely attenuated, since ωL ≪ R at these frequencies), which beat with the
fundamental to produce the 6th-harmonic torque ripple. This is the BLDC-specific failure mode
the reward-function ripple term targets (see roadmap).

**Figure 6 — Instantaneous power balance.** `fig_power_balance.png`. T_e·ω vs. Σ e_j i_j over
20,000 random operating points (states, speeds, incl. standstill and regeneration); the identity
holds to 1.8e-13 relative error.

**Figure 7 — Interface equivalence.** `fig_interface_equivalence.png`. The same clamped-PI
controller structure (per-motor gains) drives both `Cont-SC-BLDC-v0` and `Cont-SC-PMSM-v0` in
dq-action mode with identical state/action interfaces — the drop-in claim verified
end-to-end.

## 3. Notes on the numbers

- **Steady-state speed 10.00 rad/s with error −0.00**: the PI controller (kp = 1e-3,
  ki = 5e-6 per step at τ = 1e-4 s) reaches the reference with < 0.01 rad/s error and 0.06
  rad/s p-p ripple. Rise time (example script): 0.27 s.
- **Torque ripple 0.25 N·m p-p on 0.11 N·m mean**: dominated by the 5th/7th current harmonics
  (measured: i_a spectrum 1.18 A fundamental, 0.45 A 5th, 0.19 A 7th). The ideal-current-drive
  bound is 14.7% of mean; a current-controlled drive (or the future RL policy acting on the
  ripple term) can approach it.
- **J = 3e-3 kg·m²** is used for the closed-loop demonstrations (default placeholder
  j_rotor = 1e-4 is the bare rotor; the speed loop of the bare rotor is too fast for a simple
  PI to regulate smoothly — itself evidence of the control difficulty the RL approach targets).
- **Alignment factor 1.216**: the q-axis gain of the unit trapezoid, e_q = 1.216·k_e·ω.
  With k_e = 0.0955 V·s/rad this yields the effective torque constant 1.216·k_e = 0.116 N·m/A
  (consistent with the 2·k_e = 0.191 N·m/A six-step bound of §4.4 in `bldc_model.md`; the
  datasheet-derived bench range is 0.15–0.17 N·m/A to be refit at the bench).

## 4. What to check independently (reviewer / Shri checklist)

1. `python -m pytest tests/test_physical_systems/test_electric_motors.py -k Brushless tests/test_environments/test_environments.py -q`
2. `python examples/bldc/plot_bldc_verification.py` — regenerates all figures + the summary.
3. Hand-verify against Pillay & Krishnan (1989) Eqs. (3)–(6): the ODE in
   `brushless_dc_motor.py::electrical_ode` implements Eq. (7)/(7a) with R_s, L_s = L − M, and
   the (L − M) state-space form of the paper. Note the paper's model assumes the star
   connection implicitly; our Eq. (7a) makes it explicit (D8).
4. Sanity: e = k_e·ω·f at any (θ, ω); torque at standstill; the 150° alignment shift is a
   convention (D2) — the physical machine is unchanged, only the electrical angle origin.
5. Known limitations: piecewise BEMF (no magnet asymmetry), no iron losses, star connection
   with symmetric reluctance, parameters L_s/J/K_e to be measured (bldc_model.md §5–6).

## 5. Reproducibility

```
uv venv --python 3.12 .venv && uv pip install -p .venv -e .
.venv/bin/python -m pytest tests/test_physical_systems/test_electric_motors.py -k Brushless tests/test_environments/test_environments.py -q
.venv/bin/python examples/bldc/cont_sc_bldc_example.py
.venv/bin/python examples/bldc/plot_bldc_verification.py
```
