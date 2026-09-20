# BLDC PI benchmark — protocol, implementation and thesis reference

**Protocol:** `bldc-pi-v1` · **Date:** 2026-09-15 · **Scope:** simulation baseline and reusable evaluation infrastructure.

This benchmark evaluates the existing GEM BLDC plant with a discrete PI speed controller.
It creates reproducible evidence for subsequent RL comparisons. It does **not** establish
an optimally tuned classical baseline, motor-parameter accuracy, FPGA feasibility, or real-motor performance.
The simulator is GEM; the animated viewer replays GEM traces and does not use MuJoCo.

## 1. Files and reproduction

Run all commands from the `gym-electric-motor` repository root:

```bash
MPLCONFIGDIR=/private/tmp/fyp-mpl .venv/bin/python -m pytest tests/test_bldc_benchmark.py -q
MPLCONFIGDIR=/private/tmp/fyp-mpl .venv/bin/python -m benchmarks.bldc.run --output results/bldc/my-pi-run
```

Use a **new output directory** each time. Existing directories are rejected, preserving prior evidence.
`MPLCONFIGDIR` only selects a writable plotting cache; choose an appropriate local path on other operating systems.
Dependencies are those in the local GEM checkout plus pytest for tests; no RL framework or browser library is needed.
Python and package versions are recorded per run. The local project must be installed editable (`uv pip install -p .venv -e .`).

Optional selection/repeatability runs:

```bash
.venv/bin/python -m benchmarks.bldc.run --scenarios startup --seeds 0 1 --output results/bldc/startup-repeat
```

The plant, references and initial conditions here are deterministic. Different seeds are recorded for
future compatibility but are **not independent robustness trials**. Domain randomization is outside this step.

| Artifact | Contents |
|---|---|
| `config.json` | Actual selected scenarios/seeds, plant, controller, solver and metric settings |
| `manifest.json` | UTC time, command, Python/platform, Git revision/dirty state, SHA-256 source/config/output fingerprints |
| `environment-packages.json` | Installed distribution versions, collected without requiring pip |
| `traces/*_seed*.csv.gz` | Full-precision, full-resolution control-step records; deterministic gzip encoding |
| `metrics.json` | Nested run and event results, completion/failure status and missing values |
| `summary.csv`, `events.csv` | Flat tables for analysis or import into a thesis table |
| `report.md` | Readable generated results and interpretation limits |
| `figures/*.png`, `*.svg` | Four-panel figures; PNG for preview, SVG for scalable thesis inclusion |
| `viewer/dist/index.html` | Self-contained offline motor animation and synchronized plots |

Open the viewer file in a browser; use the scenario selector, Play/Pause, seek slider and playback-rate selector.
No network requests are required for playback. The source template is `benchmarks/bldc/viewer.html`.
Plots retain full-resolution data; the display payload rounds values to seven decimal places.
Only scientific traces, not rounded display values, are used to compute metrics.

## 2. Requirements and verification mapping

These requirements are project-authored, not claims of compliance with an external standard.

| ID | Requirement | Implementation | Verification/evidence |
|---|---|---|---|
| B01 | Identical plant and schedules reusable across controllers | `environment.py`, `evaluate.py`, `config.json` | `test_reference_and_load_event_alignment`; config saved per run |
| B02 | PI gains, sample time, saturation and reset explicitly defined | `control.py` | `test_antiwindup_and_reset` |
| B03 | Quantify upward/downward speed responses correctly | `metrics.py::response_metrics` | Analytical first-order tests (up, down, negative target) |
| B04 | Apply signed load disturbances on exact control boundaries | `DisturbedLoad`, `schedule_value` | Acceleration and event-alignment tests |
| B05 | Record tracking, torque ripple, current and command metrics | `metrics.py`, `evaluate.py` | Known-error integral test; raw traces and generated tables |
| B06 | Do not report an unreachable response as zero rise/settling time | `response_metrics` | Unreachable-response and minimum-hold tests |
| B07 | Stop on failure, retain observed samples, and expose unvisited events | `evaluate.py`, `summarize` | Constraint-trip and nonfinite-action tests |
| B08 | Make repeated numerical runs and trace serialization reproducible | `run.py::write_trace`, manifest | Identical rollout/gzip test |
| B09 | Preserve raw evidence and protect earlier result directories | `run.py` | Output inventory, checksums, `mkdir(exist_ok=False)` |
| B10 | Keep visualization tied to recorded physics and identify assumptions | `visuals.py`, `viewer.html` | `tests/test_bldc_viewer.cjs` runtime/data-binding smoke test; scientific figure inspection |
| B11 | Provide traceable references and distinguish original decisions | This document and `references/bldc_benchmark.bib` | Source-to-claim table in §8 |

Tests reside in `tests/test_bldc_benchmark.py`. The existing motor/environment tests remain separate:

```bash
.venv/bin/python -m pytest tests/test_physical_systems/test_electric_motors.py tests/test_environments/test_environments.py -k 'Brushless or bldc' -q
```

For the full regression suite, activate the virtual environment first: the upstream example tests
spawn `python` by name. Use the headless Matplotlib backend for those examples:

```bash
source .venv/bin/activate
MPLCONFIGDIR=/private/tmp/fyp-mpl MPLBACKEND=Agg python -m pytest tests/ -q
node tests/test_bldc_viewer.cjs results/bldc/my-pi-run/viewer/dist/index.html
```

The Node check exercises the replay with a lightweight DOM/canvas stub. It verifies runtime and data
binding, **not browser rendering or responsive layout**. A connected browser was unavailable in this session.

## 3. Plant and interface

GEM supplies the drive-environment framework [R1]. The existing custom BLDC model integrates phase-variable
electrical dynamics with trapezoidal back-EMF; the phase-variable modeling approach is motivated by Pillay
and Krishnan [R2]. See [`bldc_model.md`](bldc_model.md) for the local equations and assumptions.
This benchmark does not reproduce the paper's switching-level drive experiment.

| Quantity | Benchmark value | Provenance/status |
|---|---:|---|
| Pole pairs `p` | 21 | Existing project motor configuration; 42 poles |
| Phase resistance `r_s` | 0.085 Ω | Inherited project value; phase/line measurement convention needs bench confirmation |
| Equivalent phase inductance `l_s` | 50 µH | Placeholder |
| Mechanical back-EMF constant `k_e` | 0.0955 V s/rad | Existing KV-derived assumption; calibration unresolved |
| Rotor inertia | 0.0001 kg m² | Placeholder |
| Load inertia | 0.0029 kg m² | Project-chosen assumed attached load |
| **Total inertia** | **0.003 kg m²** | Sum of the two above; not a measured propeller or rotor value |
| Base load coefficients `(a,b,c)` | `(0.01, 0.01, 0)` | Project test load; units N m, N m s/rad, N m s²/rad² |
| DC supply | 44.4 V | Ideal supply; no battery sag modeled |
| Control period `Δt` | 100 µs | 10 kHz sampled controller |
| ODE solver | `dopri5`, relative tolerance `1e-7`, absolute tolerance `1e-9` | Explicit numerical configuration, max internal steps 10,000 |
| Converter | Continuous averaged B6 bridge, zero interlocking time | GEM converter; not switching-level PWM or measured inverter timing |
| Output action bound | `sqrt(a_d²+a_q²) ≤ 0.12` | Shared benchmark restriction; dq normalization uses voltage limits returned by the plant |

The total inertia is now explicitly split into rotor and load contributions; the old demo instead
overrode the rotor parameter with a lumped inertia. Results are therefore not asserted to match that demo exactly.

At each interval `k`, the controller receives SI-valued state quantities and the current speed reference.
Its action is held during `[kΔt,(k+1)Δt]`. The trace row at `(k+1)Δt` contains the resulting state,
the reference/action/load applied during that interval, and `omega_before` from its start.
The environment observation returned after the step carries the reference for the **next** action;
the reward is evaluated against the command just applied. This prevents a one-step reference mismatch.

All runs start from zero speed, zero phase currents and zero electrical angle, with a fresh controller.
The environment retains its default speed-error reward, but reward is logged only; it is not the basis
of the benchmark ranking and is not the proposed thesis RL reward.

### Load disturbance equation (project extension)

The mechanical equation used for this experiment is

\[
J\dot\omega = T_e-T_{base}(\omega)-d_k. \tag{B1}
\]

`d_k` is a signed applied load torque (N m), held for the whole control interval. Positive values
oppose positive rotation. This is distinct from direction-dependent friction. `DisturbedLoad` delegates
the base load and its near-zero friction smoothing to GEM's `PolynomialStaticLoad`.
Disturbance changes occur between solver calls; the right-hand side does not introduce an internal time event.

## 4. Baseline controller

This is a **single-loop PI speed controller commanding q-axis voltage**, with d-axis command zero.
It is not a cascaded speed/current controller or a six-step commutation controller.
Gains carry over from the existing demonstration, with the integral gain expressed per second:

\[
e_k=\omega_k^*-\omega_k,\qquad
z^c_{k+1}=z_k+K_i\Delta t\,e_k,\qquad
v^c_k=K_p e_k+z^c_{k+1}. \tag{B2}
\]

Here `z` and `v` are normalized voltage quantities, `e` is in rad/s,
`Kp = 0.001` has units normalized voltage/(rad/s), and `Ki = 0.05` has units
normalized voltage/rad. At 100 µs this is an integral increment of `5e-6 × e` per step.

Conditional integration prevents further accumulation into saturation [R3]: retain `z_k` when
`vᶜ_k > 0.12` with positive integral increment, or `vᶜ_k < −0.12` with negative increment.
Otherwise accept the candidate. Then

\[
a_{d,k}=0,\qquad a_{q,k}=\operatorname{clip}(K_p e_k+z_{k+1},-0.12,0.12). \tag{B3}
\]

The evaluator also projects any controller's requested dq vector onto the shared radius-0.12 action disk.
It logs the request, applied action, projection count and fraction of samples at the bound.
`action_clipped_samples` counts evaluator projection only; the PI's own saturation is reflected in
`action_at_limit_fraction`. Integral state resets to zero at episode start, never at a speed/load event.

**Tuning status:** no gain search has been performed for this protocol. The fixed original gains are an
initial reproducible baseline. A thesis claiming RL superiority over classical control should also use
a properly tuned PI/cascaded baseline and report the tuning budget separately from evaluation.

## 5. Frozen scenarios

All times are seconds; every schedule is right-continuous on the control grid. Each scenario is a fresh episode.
These values and horizons are **project design choices**, not a published standard test suite.

| Scenario | Horizon | Speed command (rad/s) | Added load torque (N m) | Purpose |
|---|---:|---|---|---|
| `startup` | 2.0 | 0; 10 at 0.1 | 0 throughout | Startup response |
| `speed_steps` | 4.5 | 0; 10 at 0.1; 20 at 1.5; 5 at 3.0 | 0 throughout | Upward and downward tracking |
| `load_step` | 3.0 | 0; 10 at 0.1 | 0; +0.08 at 1.5 | Sustained disturbance rejection |
| `load_pulse` | 3.0 | 0; 10 at 0.1 | 0; +0.08 at 1.5; 0 at 1.8 | Disturbance and release recovery |

Low-speed, nominal-parameter tests are intentionally limited. They do not cover high-speed voltage limits,
reversal under load, sensor noise, delay, parameter uncertainty, inverter faults, or hardware safety.

## 6. Metrics and missing-value rules

### Tracking and control effort (project definitions)

For `N` observed post-step errors `e_k`, the right-endpoint sampled metrics are

\[
\mathrm{MAE}=\frac1N\sum|e_k|,\quad
\mathrm{RMSE}=\sqrt{\frac1N\sum e_k^2},\quad
\mathrm{IAE}=\Delta t\sum|e_k|. \tag{B4}
\]

MAE and RMSE are in rad/s; IAE is in rad. They are reported for the entire observed run and each event segment.
The pre-command state is used for response timing, not added as an extra error sample to these averages.
Normalized action effort is `mean(a_d²+a_q²)` and is **not electrical energy**.

### Event response

Event segments run from each speed/load change to the next change or planned end. An initial hold segment
is also retained. Response conventions draw on the 10–90% rise and 2% settling definitions in [R4], with
the following explicit project adaptations:

- Rise time is `t90 − t10`, using first linearly interpolated crossings of progress
  `(ω−ω_initial)/(ω_target−ω_initial)`. It applies to upward and downward commanded steps.
- Overshoot is `100 × max(0, max(progress)−1)`. The endpoint is the **commanded target**, not the final sample.
- The speed-step band is `max(0.02 × |target−initial|, 0.05 rad/s)`.
- For a load event the band is `max(0.02 × |target|, 0.05 rad/s)`; report recovery time instead of rise/overshoot.
- Settling/recovery requires all remaining samples to remain in band for at least 0.1 s.
  It is a finite-horizon observation, not proof of asymptotic stability.

A missing threshold crossing or insufficient settling window produces JSON `null` (blank CSV; — in the report),
never zero. Load events have `null` rise and overshoot. Unvisited events explicitly have `status=not_reached`.
Partial segments cannot claim successful settling or complete tail-window statistics.

### Tail-window error and ripple (project definitions)

For the last **0.2 s of each complete segment**, report mean signed speed error, speed peak-to-peak,
mean torque, torque peak-to-peak, and torque AC RMS:

\[
T_{pp}=\max(T_e)-\min(T_e),\qquad
T_{AC,rms}=\sqrt{\operatorname{mean}[(T_e-\overline T_e)^2]}. \tag{B5}
\]

The relative peak-to-peak torque value is `100 × Tpp / |mean torque|`; it is undefined below
`|mean torque| = 0.001 N m`, avoiding meaningless division near zero. Report the absolute value alongside it.
The tail window is **not automatically steady state**: consult `settled_in_observed_window`.
Segments shorter than 0.2 s or incomplete segments have `null` tail statistics. No smoothing or decimation
is applied before computing these metrics. These are control-rate sampled extrema, not guarantees
about all intra-step continuous-time peaks or PWM ripple.

### Current constraints and failure

The existing GEM constraint trips when

\[
q_I=(i_a/I_{a,lim})^2+(i_b/I_{b,lim})^2+(i_c/I_{c,lim})^2>1. \tag{B6}
\]

The current limits are read from the plant (40 A per phase in this configuration). This is a combined
squared-current constraint, **not** three independent 40 A clamps. Equality is allowed in the current code.
The benchmark reports maximum `q_I`, violating sample count, and maximum absolute phase current separately.
Voltage saturation does not guarantee the current constraint cannot be crossed.

On termination/truncation the rollout stops immediately, preserves the terminal sample, and records the cause.
Invalid controller actions, unsuccessful ODE integration, and nonfinite observations produce explicit failures.
There is no mid-scenario reset and no zero-filled remainder. Run-wide error metrics for a failed run cover
only its observed portion and must not be ranked against completed runs without considering failure.

## 7. Visualization and thesis use

The local viewer shows an illustrative 36-slot/42-pole cross-section, an outer rotor, a fixed stator,
and a rotating shaft marker. Mechanical angle is reconstructed with trapezoidal integration of recorded
speed. It is not CAD, a winding-layout specification, or an independent physics simulation.
Playback scales time only. The current scope displays a 40 ms window; overview plots use per-pixel
min/max reduction to preserve visible extrema. Play/pause, seek, restart and experiment selection all
operate on the same recorded time axis. The initial scene is paused.

Suggested faculty walkthrough:

1. Show `startup`: identify reference, shaft response and current transient.
2. Switch to `speed_steps`: explain upward/downward response and command saturation.
3. Show `load_step`: pause near 1.5 s, then observe speed deviation and PI recovery.
4. Show `load_pulse`: distinguish disturbance application from its removal.
5. Use the generated table for numerical claims; use SVG figures for the written thesis.

Suggested methods wording (adapt to the thesis):

> A deterministic benchmark was implemented around the GEM BLDC environment. A discrete PI speed
> controller produced bounded q-axis voltage commands at a 100 µs control interval. Identical
> reference and disturbance schedules were evaluated from fixed initial conditions. Full-resolution
> traces were retained, and tracking error, step-response characteristics, sampled current-constraint
> violations, and torque variation were calculated using the stated finite-horizon definitions.

Suggested figure caption:

> Simulated BLDC response under the initial PI baseline for [scenario], seed [seed]. Panels show
> reference/shaft speed, electromagnetic and added load torque, phase currents, and normalized q-axis
> command. Parameters and gains follow protocol `bldc-pi-v1`; raw traces and provenance accompany the figure.

### Outstanding issues inherited from earlier work

- The older verification figure text refers to a 210° alignment while the implemented motor applies a
  150° shift. This benchmark uses the implemented motor unchanged and does not reproduce that alignment figure.
  Resolve the old figure's extra-shift convention before citing it in the thesis.
- The old demo labels a time-to-90% measurement as 10–90% rise time. Use this benchmark's corrected definition.
- Inductance, inertia, resistance conventions and back-EMF/torque calibration require bench identification.
- An ideal averaged bridge and this command bound do not model a complete real inverter/current-protection chain.

## 8. References and source-to-claim mapping

Bibliography: [`references/bldc_benchmark.bib`](references/bldc_benchmark.bib). Web references checked 2026-09-15.

| ID | Source and supported claim | What is not attributed to it |
|---|---|---|
| R1 | Traue, Book, Kirchgässner & Wallscheid, *Towards a Reinforcement Learning Environment Toolbox for Intelligent Electric Motor Control*, arXiv:1910.09434 (2019). [Author manuscript](https://arxiv.org/abs/1910.09434). GEM framework context. | This custom BLDC extension, our gains, or the benchmark results |
| R2 | Pillay & Krishnan, *Modeling, simulation, and analysis of permanent-magnet motor drives. II. The brushless DC motor drive*, IEEE Trans. Industry Applications 25(2), 274–279 (1989), DOI [10.1109/28.25542](https://doi.org/10.1109/28.25542). [Publisher record](https://ieeexplore.ieee.org/document/25542/). Phase-variable BLDC modeling. | Our parameter estimates, load schedule, exact equation numbering, or numerical reproduction of their experiment |
| R3 | MathWorks, [Anti-Windup Control Using PID Controller Block](https://www.mathworks.com/help/simulink/slref/anti-windup-control-using-a-pid-controller.html). Conditional integration/clamping concept. | Our discrete update order, gains, voltage bound or tuning optimality |
| R4 | MathWorks, [stepinfo](https://www.mathworks.com/help/control/ref/dynamicsystem.stepinfo.html). Conventional rise/settling characteristics. | Our absolute band floor, minimum hold, target-based endpoint, sampled windows or failure rules |

Equations (B1)–(B6) are **local document numbering**. No unverified source equation numbers are asserted.
MAE/RMSE/IAE, ripple statistics, horizons and visualization geometry are explicitly defined project choices.
Source citations support methods, not experimental outcomes; outcomes are supported by the accompanying data.
