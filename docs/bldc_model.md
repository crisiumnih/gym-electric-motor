# BLDC Motor Model — Equations and Implementation Reference

**Scope:** Single source of truth for the equations implemented in `gym-electric-motor` for the
brushless DC (BLDC) motor model. This document keeps the notation, the equations, and the code in
lockstep so that the thesis can cite one consistent set. Section 3 documents the reference
implementation (PMSM) that the BLDC model mirrors in structure and interface.

**Status:** v1 — implementation complete, tests passing. Equation numbers in this document are the
ones to use in the thesis; citations to the underlying papers are given per equation.

---

## 1. Notation

| Symbol | Unit | Description |
|---|---|---|
| $u_j$ | V | Phase voltage of phase $j \in \{a, b, c\}$ |
| $i_j$ | A | Phase current of phase $j$ |
| $e_j$ | V | Phase back-EMF of phase $j$ |
| $f_j$ | 1 | Unit back-EMF shape function of phase $j$, $f_j \in [-1, 1]$ |
| $R_s$ | $\Omega$ | Stator (phase) resistance |
| $L_s$ | H | Equivalent phase inductance $L_s = L - M$ |
| $K_e$ | V·s/rad | Back-EMF constant (mechanical) |
| $p$ | 1 | Pole pair number |
| $\epsilon$ | rad | Electrical rotor angle |
| $\omega_{\mathrm{me}}$ | rad/s | Mechanical angular velocity |
| $T_e$ | N·m | Electromagnetic torque |
| $T_L$ | N·m | Load torque |
| $J$ | kg·m² | Total moment of inertia (rotor + load) |
| $u_{\mathrm{sup}}$ | V | DC supply (bus) voltage |

GEM parameter names in code are lowercase (e.g. `r_s`, `l_s`, `k_e`, `p`, `j_rotor`) and follow
the GEM convention (lowercase for motor parameters, see
`physical_systems/electric_motors/electric_motor.py`). Units in the thesis should always be SI
(rad/s, not RPM; V·s/rad, not V/RPM).

**GEM state vector** (identical for PMSM and BLDC environments, 14 entries):

$$\underbrace{[\omega_{\mathrm{me}},\; T_e]}_{\text{mechanical}} ,\;
[\underbrace{i_a, i_b, i_c}_{3},\underbrace{i_{sd}, i_{sq}}_{2}],\;
[\underbrace{u_a, u_b, u_c}_{3},\underbrace{u_{sd}, u_{sq}}_{2}],\;
[\epsilon,\; u_{\mathrm{sup}}]$$

All entries are normalized to $[-1, 1]$ by the corresponding limit values before being handed to
the agent (BLDC: `BrushlessDCMotorSystem.simulate()` /
`physical_systems.py`).

## 2. General framework (Supply–Converter–Motor–Load)

The environment simulates a drive chain of four components (see
`physical_systems/physical_systems.py`, class `SCMLSystem`): a voltage supply, a power-electronic
converter, the electric motor, and a mechanical load. The continuous-time model is a coupled ODE
system that is integrated with an ODE solver over the control interval $\tau$:

**Electrical subsystem (motor ODE):**
$$\frac{\mathrm{d}x}{\mathrm{d}t} = f(x(t),\, u_{\mathrm{in}}(t),\, \omega_{\mathrm{me}}(t)), \qquad x \in \mathbb{R}^{n_{\mathrm{el}}}$$
(1)

with motor state $x$, phase voltage input $u_{\mathrm{in}}$ and mechanical speed $\omega_{\mathrm{me}}$
(GEM: `ElectricMotor.electrical_ode(state, u_in, omega)`).

**Torque law:** algebraic function of the motor state
$$T_e = \tau_{\mathrm{el}}(x)$$
(2)

(GEM: `ElectricMotor.torque(state)`).

**Mechanical subsystem (load ODE):** the load is driven by the motor torque
$$\frac{\mathrm{d}\omega_{\mathrm{me}}}{\mathrm{d}t} = \frac{T_e - T_L(\omega_{\mathrm{me}})}{J}$$
(3)

with a polynomial load torque $T_L(\omega_{\mathrm{me}}) = \operatorname{sign}(\omega_{\mathrm{me}})
(c\,\omega_{\mathrm{me}}^2 + b\,|\omega_{\mathrm{me}}| + a)$ (GEM:
`physical_systems/mechanical_loads/polynomial_static_load.py`). The ODE solver integrates the
concatenated system (`SCMLSystem._system_equation`), while the converter imposes the input
voltages $u_{\mathrm{in}}$ from the action and the current draw feeds back to the supply
(`SCMLSystem.simulate`). For the three-phase drives, the converter is a symmetric B6 bridge with
phase voltage range $[-u_{\mathrm{sup}}/2,\; +u_{\mathrm{sup}}/2]$ and one-sample dead time.

## 3. Reference: the PMSM as implemented in GEM

The PMSM is the reference implementation the BLDC model mirrors (same state layout, same
observers, same converter interface). Its equations are documented in
`docs/parts_gem/technicalbackground.rst` (PMSM section) and implemented in
`physical_systems/electric_motors/permanent_magnet_synchronous_motor.py`.

**PMSM dq equations** (GEM documentation, with $\omega = p\,\omega_{\mathrm{me}}$):

$$u_{sd} = R_s i_{sd} + L_d \frac{\mathrm{d}i_{sd}}{\mathrm{d}t} - \omega_{\mathrm{me}} p\, L_q i_{sq}$$
(4a)

$$u_{sq} = R_s i_{sq} + L_q \frac{\mathrm{d}i_{sq}}{\mathrm{d}t} + \omega_{\mathrm{me}} p\, L_d i_{sd} + \omega_{\mathrm{me}} p\, \Psi_p$$
(4b)

$$T_e = \frac{3}{2} p\, \bigl(\Psi_p + (L_d - L_q)\, i_{sd}\bigr)\, i_{sq}$$
(5)

with $\Psi_p$ the permanent rotor flux linkage. Implementation notes:

- The ODE states are $[i_{sd}, i_{sq}, \epsilon]$ (`synchronous_motor.py`, `CURRENTS_IDX = [0, 1]`,
  `EPSILON_IDX = 2`), and $\frac{\mathrm{d}\epsilon}{\mathrm{d}t} = p\,\omega_{\mathrm{me}}$
  (third row of `_model_constants` in `permanent_magnet_synchronous_motor.py`).
- The ODE is written as a linear matrix `_model_constants` times
  $[\omega_{\mathrm{me}}, i_{sd}, i_{sq}, u_{sd}, u_{sq}, \omega_{\mathrm{me}} i_{sd},
  \omega_{\mathrm{me}} i_{sq}]$, divided by $L_d$/$L_q$ per row — see
  `PermanentMagnetSynchronousMotor._update_model()`.
- The electrical angle dynamics and the Park transform use $\epsilon = p\,\theta_{\mathrm{me}}$
  (`three_phase_motor.py`, `q`/`q_inv`).
- The BEMF of the PMSM sits on the **q-axis**: $e_q = \omega_{\mathrm{me}} p\, \Psi_p$, $e_d = 0$
  (Eq. 4b). A q-axis voltage therefore drives torque-producing current. The BLDC model is
  aligned to the same convention (Section 4.3).

## 4. The BLDC model (this work)

### 4.1 Phase-variable model

Source: **Pillay & Krishnan (1989), "Modeling, Simulation, and Analysis of Permanent-Magnet
Motor Drives, Part II: The Brushless DC Motor Drive," IEEE Trans. Ind. Appl. 25(2), pp. 274–279**
— the canonical reference. Their model, Eqs. (3)–(6) there (equation numbers to be
cross-checked against the PDF when writing the thesis):

The machine is modeled in **phase variables (abc)**. Because the back-EMF is trapezoidal, the
transformation to a dq frame "is not necessarily the best approach" for simulation (their §II
argument), and a Fourier-series dq model would require many terms. With equal phase resistances,
equal self-inductances $L$, equal mutual inductances $M$ (symmetric rotor reluctance, star
connection) the phase voltage equations are [P&K Eq. (3)]:

$$\underbrace{\begin{pmatrix} u_a \\ u_b \\ u_c \end{pmatrix}}_{u_{abc}} =
R_s \underbrace{\begin{pmatrix} i_a \\ i_b \\ i_c \end{pmatrix}}_{i_{abc}} +
L_s \frac{\mathrm{d}}{\mathrm{d}t} i_{abc} +
\underbrace{\begin{pmatrix} e_a \\ e_b \\ e_c \end{pmatrix}}_{e_{abc}}, \qquad
L_s = L - M$$
(6)

In state-space form [P&K Eq. (6)]:

$$\frac{\mathrm{d}i_{abc}}{\mathrm{d}t} =
\frac{1}{L_s}\Bigl(u_{abc} - u_n - R_s\, i_{abc} - e_{abc}\Bigr)$$
(7)

with the floating neutral potential of the star connection

$$u_n = \frac{1}{3} \sum_{j \in \{a,b,c\}} \bigl(u_j - R_s i_j - e_j\bigr)$$
(7a)

which guarantees $\frac{\mathrm{d}}{\mathrm{d}t}(i_a + i_b + i_c) = 0$ identically, i.e. the
common-mode current stays zero. This term is required in the phase-variable formulation
because the trapezoidal back-EMF has a nonzero common-mode component
($f_a + f_b + f_c \neq 0$ in general, unlike sinusoidal back-EMF); without it, the
three independent phase ODEs would accumulate a spurious circulating common-mode current.
In the P&K model the same result is implicit in the $(L - M)$ structure of their state-space
form.

The electrical angle follows the pole-pair relation [P&K Eq. (5)]:

$$\frac{\mathrm{d}\epsilon}{\mathrm{d}t} = p\,\omega_{\mathrm{me}}$$
(8)

### 4.2 Back-EMF shape function

The phase back-EMFs are [P&K §II, Fig. 1]

$$e_j(\epsilon, \omega_{\mathrm{me}}) = K_e\, \omega_{\mathrm{me}}\, f_j(\epsilon), \qquad
j \in \{a, b, c\}$$
(9)

with the phase-a shape function a unit trapezoid of amplitude $\pm 1$, flat over $2\pi/3$
electrical and linear over $\pi/3$ electrical:

$$f_a(\epsilon) = \begin{cases}
1, & 0 \le \epsilon < 2\pi/3 \\
1 - \frac{6}{\pi}\left(\epsilon - \frac{2\pi}{3}\right), & 2\pi/3 \le \epsilon < \pi \\
-1, & \pi \le \epsilon < 5\pi/3 \\
-1 + \frac{6}{\pi}\left(\epsilon - \frac{5\pi}{3}\right), & 5\pi/3 \le \epsilon < 2\pi
\end{cases}$$
(10)

and $f_b(\epsilon) = f_a(\epsilon - 2\pi/3)$, $f_c(\epsilon) = f_a(\epsilon - 4\pi/3)$.
This is the piecewise-linear form used in the paper; the Fourier-series alternative is
explicitly avoided there (Gibbs phenomenon). The implementation offers an optional numerical
smoothing parameter (`bemf_smoothing`) that blends the $\pi/3$ transitions toward a smoothstep
curve while preserving the flat plateaus exactly; with `0.0` (default) the trapezoid is exactly
Eq. (10). See decision D3.

### 4.3 Angular alignment (dq interface)

The electrical angle origin is a free modeling convention. In GEM the Park transform and the
PMSM place the back-EMF on the q-axis ($e_d = 0$, $e_q = \omega p\Psi_p > 0$). The BLDC trapezoid
of Eq. (10) has its fundamental component on the line $e_d = -e_q/\sqrt{3}$ in that frame; a pure
q-axis voltage would then drive almost no net torque, which breaks the intended "drop-in dq
agent" interface (decision D1).

**Convention adopted:** the phase-a shape function is shifted by $+5\pi/6$ (150° electrical),
$f_a(\epsilon) = f_{\text{Eq.(10)}}(\epsilon + 5\pi/6)$, which places the fundamental of the
back-EMF exactly on the q-axis: numerically, $(e_d, e_q) = (0,\; 1.216\, K_e \omega_{\mathrm{me}})$
over one electrical cycle (verified by test `test_BrushlessDCMotor_backemf_shape`). The physical
machine is unchanged — only the definition of the zero angle is re-aligned to the dq frame.

Implementation: `BrushlessDCMotor.bemf_shape()` and the `+5π/6` shift in
`_bemf_phase_a()` (`brushless_dc_motor.py`).

### 4.4 Torque

From the instantaneous power balance [P&K Eq. (4)]

$$T_e = \frac{e_a i_a + e_b i_b + e_c i_c}{\omega_{\mathrm{me}}}$$
(11)

which with Eq. (9) becomes the form implemented (well defined at standstill, no division by
$\omega_{\mathrm{me}}$):

$$T_e = K_e \sum_{j \in \{a,b,c\}} f_j(\epsilon)\, i_j$$
(12)

**Important convention check for the thesis:** with two phases conducting in six-step operation
($i_a = I$, $i_b = -I$, $f_a = f_b = \pm 1$), Eq. (12) gives $T_e = 2 K_e I$. The effective
torque constant per line current is therefore $2 K_e$, not $K_e$. With $K_e = 1/KV$ from the
datasheet ($0.0955$ V·s/rad), $2 K_e = 0.191$ N·m/A, consistent with the order of magnitude of
the manufacturer's bench data (~$0.15$–$0.17$ N·m/A; see Section 5).

### 4.5 Mechanical dynamics

Identical to the generic framework, Eq. (3), with $J = j\_rotor + J_{\text{load}}$ and the
polynomial load torque. The BLDC-specific behaviour enters only through the 6th-harmonic torque
ripple produced by Eq. (12) with the trapezoidal shape — the physical ripple that motivates the
ripple term of the reward function (see `roadmap.md`).

### 4.6 Implementation mapping

| Equation | Code | File |
|---|---|---|
| (6), (7), (7a) phase ODEs | `electrical_ode()` | `electric_motors/brushless_dc_motor.py` |
| (8) angle dynamics | last row of `electrical_ode()` | same |
| (9) back-EMF | `back_emf()` | same |
| (10) shape function | `bemf_shape()`, `_bemf_phase_a()` | same |
| (12) torque | `torque()` | same |
| (3) mechanical ODE | `mechanical_ode()` | `physical_systems/mechanical_loads/` |
| state vector, normalization | `BrushlessDCMotorSystem.simulate()/reset()` | `physical_systems/physical_systems.py` |
| dq→abc action mapping | `dq_to_abc_space()` (inherited) | `ThreePhaseMotorSystem`, same file |
| env (speed control, dq action) | `ContSpeedControlBrushlessDCMotorEnv` | `envs/gym_bldc/cont_sc_bldc_env.py`, id `Cont-SC-BLDC-v0` |

Motor ODE state layout (class attributes of `BrushlessDCMotor`):
`CURRENTS = ['i_a','i_b','i_c']`, `CURRENTS_IDX = [0,1,2]`, `EPSILON_IDX = 3`, `VOLTAGES =
['u_a','u_b','u_c']`. The observed dq quantities (`i_sd`, `i_sq`, `u_sd`, `u_sq`) are derived from
the abc quantities via the Clarke/Park transform in the physical system and are **observation-only**
— they do not appear in the ODE.

## 5. Motor parameters — T-Motor Antigravity KV100 (36N42P)

Defaults of `BrushlessDCMotor` (all overridable via the `motor_parameter` dict at env
construction).

| Parameter | Symbol | Value | Unit | Source / status |
|---|---|---|---|---|
| pole pairs | $p$ | 21 | 1 | datasheet (36N42P) — **verified** |
| phase resistance | $R_s$ | 85e-3 | Ω | datasheet — **verified** |
| equivalent inductance | $L_s$ | 50e-6 | H | **placeholder** — to be measured (locked-rotor test, see roadmap W3) |
| back-EMF constant | $K_e$ | 0.0955 | V·s/rad | $1/KV$ from datasheet (KV 100 RPM/V); **to be refit** from bench data (datasheet torque implies ≈0.15–0.17 N·m/A effective, i.e. $K_e$ ≈ 0.075–0.085, see §4.4) |
| rotor inertia | $J_{\mathrm{rotor}}$ | 1e-4 | kg·m² | **placeholder** — not published by T-Motor; to be identified |
| BEMF smoothing | — | 0.0 | 1 | numerical parameter, not physical (D3) |
| speed limit | $\omega_{\mathrm{lim}}$ | 465 | rad/s | 44.4 V · KV 100 = 4440 RPM no-load |
| current limit | $i_{\mathrm{lim}}$ | 40 | A | datasheet (180 s rating) |
| supply voltage | $u_{\mathrm{sup}}$ | 44.4 | V | 12S LiPo nominal |

**Domain randomization (training-time):** per the roadmap, $L_s$ is randomized per episode over
approximately one decade around the placeholder (tens of µH, e.g. log-uniform in
$[10, 100]\,\mu$H) and $K_e$ over the KV-derived vs. bench-fit range, until the measured values
are available. **$J$ must be randomized as well** — it is the most uncertain parameter (bare
rotor ≈ 1e-4 vs. propeller-loaded ≈ 1e-3…1e-2 kg·m², two orders of magnitude), and a policy
trained against a single inertia will not transfer to the bench. Recommended: log-uniform in
$[1e-4, 3e-3]$ kg·m² (bare rotor to propeller-class load) until an identification test
(spin-up) provides a measured value. This is handled by the training harness, not by the motor
class.

## 6. Assumptions & open items

| # | Assumption | Consequence if wrong |
|---|---|---|
| A1 | Star-connected stator, symmetric rotor reluctance ($L_a = L_b = L_c$, equal mutuals) | $L_s = L-M$ model exact; otherwise dq/abc mutual coupling terms appear |
| A2 | No damper windings, no eddy currents (P&K §II) | high-frequency losses not modeled |
| A3 | Ideal trapezoidal BEMF (no rounding, no asymmetry) | real motors deviate slightly; smoothing option exists (D3) |
| A4 | 150° alignment convention (D2) | changes nothing physical; only the dq frame origin |
| A5 | Torque from power balance Eq. (11) at all speeds | exact for the model; measurement of $K_e$ on bench decides the effective constant |
| A6 | $i_c = -i_a - i_b$ enforced by the floating neutral potential (Eq. 7a) | common-mode current identically zero; correct for isolated star connection |

**Open items to resolve before sim-to-real (roadmap W10):** measure $L_s$ (locked-rotor AC test),
refit $K_e$ (or the effective $2K_e$) from the bench data, identify $J$ (spin-up test), verify the
BEMF shape on the oscilloscope (optional: set `bemf_smoothing` if the ODE solver shows
stiffness-induced oscillation at high speed).

## 7. Decisions log

| # | Decision | Rationale |
|---|---|---|
| D1 | abc-frame plant + dq action/observer interface | authentic trapezoidal BEMF (P&K §II); state vector byte-identical to the PMSM env → RL pipeline (observer, reward, baselines) works unchanged on both envs |
| D2 | 150° alignment of the trapezoid (Section 4.3) | fundamental of the BEMF on the +q axis, $e_d = 0$; q-voltage drives torque like in the PMSM |
| D3 | sharp trapezoid default + optional `bemf_smoothing` | paper-faithful by default; smoothing as documented numerical deviation to help stiff solvers |
| D4 | torque via Eq. (12), not $e\cdot i/\omega$ | identical algebraically, well-defined at standstill |
| D5 | default parameters = Antigravity KV100 | this project's motor; placeholders explicitly flagged (Section 5) |
| D6 | env default `control_space='dq'` | roadmap decision (continuous dq voltage action for the RL/FPGA pipeline); `abc` available |
| D7 | `calc_jacobian=False` in the env | no analytic Jacobian for the piecewise BEMF; solver runs without it |
| D8 | star connection enforced via floating neutral potential (Eq. 7a) | the trapezoid has nonzero common-mode back-EMF; without the neutral term the three independent phase ODEs accumulate a spurious common-mode current that inflates the torque ripple and the phase currents (found in verification, fixed in `electrical_ode`) |

## 8. References

1. P. Pillay and R. Krishnan, "Modeling, simulation, and analysis of permanent-magnet motor
   drives, Part II: The brushless DC motor drive," *IEEE Trans. Ind. Appl.*, vol. 25, no. 2,
   pp. 274–279, 1989.
2. R. Krishnan, *Electric Motor Drives: Modeling, Analysis, and Control*. Prentice Hall, 2001
   (BLDC chapter; secondary source for Eqs. (6)–(12)).
3. A. Traue, G. Book, W. Kirchgässner, and O. Wallscheid, "Towards a reinforcement learning
   environment toolbox for intelligent electric motor control," arXiv:1910.09434 (GEM paper;
   framework of Section 2).
