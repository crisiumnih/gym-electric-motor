# BLDC benchmark

From the repository root:

```bash
MPLCONFIGDIR=/private/tmp/fyp-mpl .venv/bin/python -m benchmarks.bldc.run --output results/bldc/new-run
```

Open `results/bldc/new-run/viewer/dist/index.html` for the offline animated replay.
Start with [the protocol](../../docs/bldc_benchmark.md) for equations, metric definitions,
assumptions, citations, reproduction instructions, and the requirements/test matrix.

- `config.json`: frozen nominal scenarios and settings.
- `control.py`: initial PI baseline; future policies implement `reset()` and `act(state_si, reference_rad_s, dt)`.
- `environment.py`, `evaluate.py`: shared plant and rollout.
- `metrics.py`: controller-independent evaluation.
- `run.py`: experiment orchestration and evidence export.
- `visuals.py`, `viewer.html`: scientific figures and offline replay.

The initial PI gains are inherited from the
earlier demonstration; this is an initial baseline, not a claim of optimal classical control.

## RL training and FPGA boundary

The separate `rl/` package adds nominal SB3 training, shared observation/action contracts,
actor-only float32 export and comparison through the existing evaluator. See
[architecture.md](../../architecture.md) and [the RL protocol](../../docs/bldc_rl_protocol.md).
No MuJoCo physics, quantization or FPGA synthesis is included in this stage.

## Cascaded PI comparison

The new `pi_cascaded` controller has separately validated current/speed loops. See
[its design and tuning record](../../docs/bldc_cascaded_pi.md), including limits on hardware transfer.

```bash
.venv/bin/python -m benchmarks.bldc.run --config benchmarks/bldc/config_cascaded.json --compare-initial --output results/bldc/new-comparison
```

The resulting replay includes both controllers, and paired figures compare actual current waveforms,
speed responses and torque ripple. The original benchmark artifacts are preserved.
