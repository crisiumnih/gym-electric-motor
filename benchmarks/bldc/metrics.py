"""Sampled, finite-horizon metrics. Definitions and references: docs/bldc_benchmark.md."""

import numpy as np


def crossing(t, progress, level):
    indices = np.flatnonzero(progress >= level)
    if not len(indices):
        return None
    i = indices[0]
    if i == 0:
        return float(t[0])
    weight = (level - progress[i - 1]) / (progress[i] - progress[i - 1])
    return float(t[i - 1] + weight * (t[i] - t[i - 1]))


def response_metrics(t, omega, target, initial, band, minimum_hold, speed_step=True):
    """t starts at the event; includes the pre-action state at t=0.

    Settling is the first in-band sample after the last excursion, provided at least
    minimum_hold seconds remain. No asymptotic/stability conclusion is implied.
    """
    t, omega = np.asarray(t), np.asarray(omega)
    error = target - omega
    outside = np.flatnonzero(np.abs(error) > band)
    start = int(outside[-1] + 1) if len(outside) else 0
    settled = start < len(t) and t[-1] - t[start] >= minimum_hold - 1e-12
    result = {
        "settling_or_recovery_s": float(t[start]) if settled else None,
        "settled_in_observed_window": bool(settled),
        "band_rad_s": float(band),
        "peak_abs_error_rad_s": float(np.max(np.abs(error))),
        "rise_10_90_s": None,
        "overshoot_pct": None,
    }
    delta = target - initial
    if speed_step and abs(delta) > 1e-12:
        progress = (omega - initial) / delta
        t10, t90 = crossing(t, progress, 0.1), crossing(t, progress, 0.9)
        result["rise_10_90_s"] = t90 - t10 if t10 is not None and t90 is not None else None
        result["overshoot_pct"] = float(max(0.0, np.max(progress) - 1) * 100)
    return result


def tracking_metrics(error, dt):
    error = np.asarray(error)
    return {
        "mae_rad_s": float(np.mean(np.abs(error))),
        "rmse_rad_s": float(np.sqrt(np.mean(error**2))),
        "iae_rad": float(np.sum(np.abs(error)) * dt),
    }


def current_quality(trace, selection):
    """Angle-domain selected harmonic ratio, not an FFT/full-band THD estimate."""
    angle = trace["epsilon"][selection]
    if len(angle) < 30 or np.ptp(np.unwrap(angle)) < 2 * np.pi:
        return {"tail_phase_a_harmonic_ratio_pct": None}
    columns = [np.ones(len(angle))]
    for harmonic in (1, 3, 5, 7, 9, 11, 13):
        columns.extend([np.sin(harmonic * angle), np.cos(harmonic * angle)])
    coefficients = np.linalg.lstsq(np.column_stack(columns), trace["i_a"][selection], rcond=None)[0]
    amplitudes = np.hypot(coefficients[1::2], coefficients[2::2])
    ratio = 100 * np.linalg.norm(amplitudes[1:]) / amplitudes[0] if amplitudes[0] >= .001 else None
    return {"tail_phase_a_harmonic_ratio_pct": float(ratio) if ratio is not None else None}


def summarize(trace, scenario, config, status):
    dt, options = config["tau_s"], config["metrics"]
    n = len(trace["time_s"])
    summary = dict(status)
    summary.update(scenario=scenario["name"], samples=n, planned_samples=round(scenario["duration_s"] / dt))
    if not n:
        summary["events"] = []
        return summary
    summary.update(tracking_metrics(trace["reference_rad_s"] - trace["omega"], dt))
    summary.update(
        observed_duration_s=float(trace["time_s"][-1]),
        peak_phase_current_a=float(np.max(np.abs(np.column_stack([trace[k] for k in ("i_a", "i_b", "i_c")])))),
        max_current_constraint_ratio=float(np.max(trace["current_constraint_ratio"])),
        constraint_violating_samples=int(np.sum(trace["current_constraint_ratio"] > 1)),
        action_at_limit_fraction=float(np.mean(trace["action_at_limit"])),
        action_clipped_samples=int(np.sum(trace["action_clipped"])),
        action_effort=float(np.mean(trace["action_d"]**2 + trace["action_q"]**2)),
    )
    if "i_q_reference_a" in trace:
        summary["dq_current_tracking_rmse_a"] = float(np.sqrt(np.mean(
            (trace["i_sd"] - trace["i_d_reference_a"])**2 + (trace["i_sq"] - trace["i_q_reference_a"])**2)))
        summary["voltage_saturation_fraction"] = float(np.mean(trace["voltage_saturated"]))
        summary["current_reference_limit_fraction"] = float(np.mean(trace["current_reference_limited"]))
    events = sorted({round(t / dt) for key in ("reference", "disturbance") for t, _ in scenario[key]})
    planned = round(scenario["duration_s"] / dt)
    rows = []
    for pos, start in enumerate(events):
        end = events[pos + 1] if pos + 1 < len(events) else planned
        if start >= n:
            rows.append({"event_time_s": start * dt, "status": "not_reached"})
            continue
        stop = min(end, n)
        complete = stop == end and (status["completed"] or stop < n)
        target = float(trace["reference_rad_s"][start])
        initial = float(trace["omega_before"][start])
        previous_ref = trace["reference_rad_s"][start - 1] if start else initial
        speed_step = abs(target - previous_ref) > 1e-12
        scale = abs(target - initial) if speed_step else abs(target)
        band = max(options["settling_fraction"] * scale, options["absolute_band_rad_s"])
        t = np.arange(stop - start + 1) * dt
        omega = np.r_[initial, trace["omega"][start:stop]]
        row = {
            "event_time_s": start * dt,
            "event_end_s": end * dt,
            "status": "complete" if complete else "partial",
            "kind": "speed_step" if speed_step else ("initial_hold" if not start else "load_change"),
            "reference_rad_s": target,
            "disturbance_nm": float(trace["disturbance_nm"][start]),
        }
        row.update(tracking_metrics(target - omega[1:], dt))
        row.update(response_metrics(t, omega, target, initial, band, options["minimum_hold_s"], speed_step))
        if not complete:
            row["settling_or_recovery_s"] = None
            row["settled_in_observed_window"] = False
        tail_count = round(options["tail_window_s"] / dt)
        row.update(tail_mean_error_rad_s=None, tail_speed_pp_rad_s=None, tail_torque_mean_nm=None,
                   tail_torque_pp_nm=None, tail_torque_ac_rms_nm=None, tail_torque_pp_pct=None,
                   tail_phase_a_harmonic_ratio_pct=None, tail_dq_current_tracking_rmse_a=None)
        if complete and stop - start >= tail_count:
            tail = slice(stop - tail_count, stop)
            torque = trace["torque"][tail]
            mean = float(np.mean(torque))
            pp = float(np.ptp(torque))
            row.update(
                tail_mean_error_rad_s=float(np.mean(target - trace["omega"][tail])),
                tail_speed_pp_rad_s=float(np.ptp(trace["omega"][tail])),
                tail_torque_mean_nm=mean,
                tail_torque_pp_nm=pp,
                tail_torque_ac_rms_nm=float(np.std(torque)),
                tail_torque_pp_pct=100 * pp / abs(mean) if abs(mean) >= options["torque_mean_floor_nm"] else None,
            )
            row.update(current_quality(trace, tail))
            if "i_q_reference_a" in trace:
                row["tail_dq_current_tracking_rmse_a"] = float(np.sqrt(np.mean(
                    (trace["i_sd"][tail] - trace["i_d_reference_a"][tail])**2
                    + (trace["i_sq"][tail] - trace["i_q_reference_a"][tail])**2)))
        rows.append(row)
    summary["events"] = rows
    return summary
