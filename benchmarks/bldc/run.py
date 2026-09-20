"""Run from repo root: python -m benchmarks.bldc.run --output results/bldc/pi-v1"""

import argparse
import copy
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import platform
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from .cascaded import make_controller, derived_gains
from .evaluate import evaluate

ROOT = Path(__file__).resolve().parents[2]


def validate(config):
    dt = config["tau_s"]
    if dt <= 0 or not 0 < config["action_norm_limit"] <= 1:
        raise ValueError("Invalid sample time or action limit")
    names = set()
    if not config["scenarios"]:
        raise ValueError("At least one scenario is required")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Provide distinct seeds")
    for scenario in config["scenarios"]:
        name = scenario["name"]
        if name in names or not name.replace("_", "").isalnum():
            raise ValueError("Scenario names must be unique alphanumeric/underscore identifiers")
        names.add(name)
        duration = scenario["duration_s"]
        if duration <= 0 or not np.isclose(duration / dt, round(duration / dt), atol=1e-8, rtol=0):
            raise ValueError("Duration must be a positive multiple of tau")
        for key in ("reference", "disturbance"):
            times = [entry[0] for entry in scenario[key]]
            if not times or times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError("Schedules must start at zero and have increasing times")
            if any(t >= duration or not np.isclose(t / dt, round(t / dt), atol=1e-8, rtol=0) for t in times):
                raise ValueError("Events must lie on the control grid and before the scenario end")
            if not np.isfinite(scenario[key]).all():
                raise ValueError("Schedule values must be finite")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(config):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    files = sorted(list((ROOT / "src").rglob("*.py")) + list((ROOT / "benchmarks").rglob("*.py"))
                   + list((ROOT / "benchmarks/bldc").glob("*.html")) + list((ROOT / "benchmarks/bldc").glob("*.json")))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "python": sys.version, "platform": platform.platform(),
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "packages": {p: importlib.metadata.version(p) for p in ("gym_electric_motor", "gymnasium", "numpy", "scipy", "matplotlib")},
    }


def write_trace(path, trace):
    # Fixed gzip timestamp and empty filename make identical raw data byte-identical.
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(trace)
            writer.writerows(zip(*trace.values()))


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, summaries, config):
    lines = ["# PI benchmark results", "", f"Protocol: `{config['protocol_version']}`. Simulation only; assumed plant parameters.",
             "", "Whole-run errors include startup; partial runs must not be compared as successful full runs.", "",
             "| Controller | Scenario | Seed | Complete | RMSE (rad/s) | Peak phase current (A) | Violating samples |",
             "|---|---|---:|---|---:|---:|---:|"]
    for row in summaries:
        rmse = f"{row['rmse_rad_s']:.4f}" if "rmse_rad_s" in row else "—"
        peak = f"{row['peak_phase_current_a']:.3f}" if "peak_phase_current_a" in row else "—"
        lines.append(f"| {row['controller']} | {row['scenario']} | {row['seed']} | {row['completed']} | {rmse} | "
                     f"{peak} | {row.get('constraint_violating_samples', 'N/A')} |")
    lines += ["", "## Event responses", "", "Times are relative to each event. — means not reached, not applicable, or not settled in the window; see metrics.json.", "",
              "| Scenario | Event (s) | Type | Rise 10–90% (s) | Settle/recover (s) | Overshoot (%) | Tail mean error (rad/s) | Torque p-p (N m) |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "—" if value is None else f"{value:.4f}"
    for result in summaries:
        for row in result["events"]:
            lines.append(f"| {result['controller']} / {result['scenario']} / seed {result['seed']} | {row['event_time_s']:.2f} | {row.get('kind', row['status'])} | "
                         + " | ".join(fmt(row.get(k)) for k in ("rise_10_90_s", "settling_or_recovery_s", "overshoot_pct", "tail_mean_error_rad_s", "tail_torque_pp_nm")) + " |")
    lines += ["", "## Current quality and torque ripple", "",
              "Selected phase-a harmonic ratio uses orders 3,5,7,9,11,13 relative to the fundamental; it is not full-band THD.", "",
              "| Controller / scenario | Event (s) | Selected harmonic ratio (%) | dq tracking RMSE (A) | Torque AC RMS (N m) |",
              "|---|---:|---:|---:|---:|"]
    for result in summaries:
        for event in result["events"]:
            if event.get("tail_torque_ac_rms_nm") is not None:
                lines.append(f"| {result['controller']} / {result['scenario']} | {event['event_time_s']:.2f} | "
                             + " | ".join(fmt(event.get(k)) for k in ("tail_phase_a_harmonic_ratio_pct", "tail_dq_current_tracking_rmse_a", "tail_torque_ac_rms_nm")) + " |")
    lines += ["", "## Interpretation limits", "", "- Tail windows are 0.2 s; use the settled flag before describing them as steady state.",
              "- Current checks use GEM's sum of squared normalized phase currents, not just a per-phase threshold.",
              "- Deterministic scenarios and fixed initial states do not establish robustness or real-motor performance.",
              "- See the repository's `docs/bldc_benchmark.md` for definitions, references, and assumptions.",
              "- Open `viewer/dist/index.html` for synchronized replay. Raw measurements remain in `traces/*.csv.gz`.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--compare-initial", action="store_true", help="Run the original voltage PI with identical plant/scenarios as a paired comparison")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.scenarios:
        known = {s["name"] for s in config["scenarios"]}
        if set(args.scenarios) - known:
            parser.error("Unknown scenario")
        config["scenarios"] = [s for s in config["scenarios"] if s["name"] in args.scenarios]
    if args.seeds is not None:
        config["seeds"] = args.seeds
    designs = [config["controller"]]
    if args.compare_initial:
        if config["controller"]["name"] == "pi_initial":
            parser.error("Select a different controller config for --compare-initial")
        initial = json.loads(Path(__file__).with_name("config.json").read_text())["controller"]
        designs = [initial, config["controller"]]
        config["comparison_controllers"] = designs
    validate(config)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "traces").mkdir()
    metadata = provenance(config)
    if config["controller"]["name"] == "pi_cascaded":
        metadata["derived_gains"] = derived_gains(config, config["controller"])
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    packages = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    (args.output / "environment-packages.json").write_text(json.dumps(dict(packages), indent=2))
    summaries, runs = [], []
    for scenario in config["scenarios"]:
        for seed in config["seeds"]:
            for design in designs:
                run_config = copy.deepcopy(config)
                run_config["controller"] = design
                controller = make_controller(run_config)
                trace, summary = evaluate(run_config, scenario, controller, seed)
                summary["controller"] = design["name"]
                name = f"{scenario['name']}_seed{seed}"
                if len(designs) > 1:
                    name = f"{design['name']}_{name}"
                write_trace(args.output / "traces" / f"{name}.csv.gz", trace)
                summaries.append(summary)
                runs.append((name, trace, summary))
                (args.output / "metrics.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
                print(f"{design['name']} / {name}: completed={summary['completed']}, RMSE={summary.get('rmse_rad_s')}, failure={summary['failure']}", flush=True)
    write_csv(args.output / "summary.csv", [{k: v for k, v in r.items() if k != "events"} for r in summaries])
    write_csv(args.output / "events.csv", [dict(controller=r["controller"], scenario=r["scenario"], seed=r["seed"], **event) for r in summaries for event in r["events"]])
    write_report(args.output, summaries, config)
    from .visuals import export_visuals
    export_visuals(args.output, config, runs)
    metadata["artifacts_sha256"] = {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    if not all(r["completed"] for r in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
