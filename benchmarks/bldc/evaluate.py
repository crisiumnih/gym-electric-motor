"""Controller-independent rollout; no resets or zero padding after failure."""

import numpy as np

from .environment import make_environment, schedule_value
from .metrics import summarize
from .interfaces import project_voltage


def evaluate(config, scenario, controller, seed):
    env, load = make_environment(config, scenario)
    ps = env.unwrapped.physical_system
    names, limits = list(ps.state_names), np.asarray(ps.limits)
    current_indices = [names.index(k) for k in ("i_a", "i_b", "i_c")]
    dt = config["tau_s"]
    rows = []
    status = {"seed": seed, "completed": False, "terminated": False, "truncated": False, "failure": None}
    try:
        observation, _ = env.reset(seed=seed)
        controller.reset()
        for step in range(round(scenario["duration_s"] / dt)):
            state = dict(zip(names, np.asarray(observation[0]) * limits))
            reference = float(observation[1][0] * limits[names.index("omega")])
            load.disturbance_nm = schedule_value(scenario["disturbance"], step, dt)
            requested = np.asarray(controller.act(state, reference, dt), dtype=float)
            norm = np.linalg.norm(requested)
            action = project_voltage(requested, config["action_norm_limit"])
            observation, reward, terminated, truncated, _ = env.step(action)
            normalized = np.asarray(observation[0])
            if not np.isfinite(normalized).all() or not np.isfinite(reward):
                raise RuntimeError("Nonfinite environment observation/reward")
            row = dict(zip(names, normalized * limits))
            row.update(
                time_s=(step + 1) * dt, reference_rad_s=reference,
                omega_before=state["omega"], disturbance_nm=load.disturbance_nm,
                requested_d=requested[0], requested_q=requested[1], action_d=action[0], action_q=action[1],
                action_clipped=float(norm > config["action_norm_limit"] + 1e-12),
                action_at_limit=float(np.linalg.norm(action) >= config["action_norm_limit"] - 1e-12),
                reward=float(reward), current_constraint_ratio=float(np.sum(normalized[current_indices]**2)),
                terminated=float(terminated), truncated=float(truncated),
            )
            diagnostics = getattr(controller, "diagnostics", {})
            if any(key in row for key in diagnostics):
                raise ValueError("Controller diagnostic names must not overwrite plant trace fields")
            if not all(np.isfinite(value) for value in diagnostics.values()):
                raise ValueError("Controller diagnostics must be finite")
            row.update(diagnostics)
            rows.append(row)
            if terminated or truncated:
                status.update(terminated=bool(terminated), truncated=bool(truncated),
                              failure="current_constraint" if terminated else "environment_truncation")
                break
        else:
            status["completed"] = True
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        status["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    trace = {key: np.array([row[key] for row in rows]) for key in rows[0]} if rows else {"time_s": np.array([])}
    return trace, summarize(trace, scenario, config, status)
