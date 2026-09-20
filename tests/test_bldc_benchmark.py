"""Numerical definitions, disturbance timing, reproducibility, and failure semantics."""

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.bldc.control import PIController
from benchmarks.bldc.environment import DisturbedLoad, make_environment, schedule_value
from benchmarks.bldc.evaluate import evaluate
from benchmarks.bldc.metrics import response_metrics, tracking_metrics
from benchmarks.bldc.run import validate, write_trace


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "benchmarks/bldc/config.json").read_text())


@pytest.mark.parametrize("initial,target", [(0.0, 10.0), (20.0, 5.0), (0.0, -10.0)])
def test_first_order_response_matches_analytic_times(initial, target):
    tau = .1
    t = np.arange(0, 1, .0001)
    omega = target + (initial - target) * np.exp(-t / tau)
    result = response_metrics(t, omega, target, initial, .02 * abs(target - initial), .1)
    assert result["rise_10_90_s"] == pytest.approx(tau * np.log(9), abs=1e-6)
    assert result["settling_or_recovery_s"] == pytest.approx(-tau * np.log(.02), abs=.0001)
    assert result["overshoot_pct"] == 0


def test_settling_requires_last_excursion_and_minimum_hold():
    t = np.arange(6) * .1
    result = response_metrics(t, [0, 1, 1, .8, 1, 1], 1, 0, .02, .1)
    assert result["settling_or_recovery_s"] == pytest.approx(.4)
    assert response_metrics(t, [0, 1, 1, .8, 1, 1], 1, 0, .02, .2)["settling_or_recovery_s"] is None


def test_unreachable_response_has_no_rise_or_settling():
    result = response_metrics([0, .5, 1], [0, .3, .5], 1, 0, .02, .1)
    assert result["rise_10_90_s"] is None
    assert result["settling_or_recovery_s"] is None


def test_load_recovery_has_no_rise_or_overshoot():
    result = response_metrics([0, .1, .2, .3, .4], [10, 9, 10, 10, 10], 10, 10, .2, .1, False)
    assert result["rise_10_90_s"] is None
    assert result["overshoot_pct"] is None
    assert result["settling_or_recovery_s"] == pytest.approx(.2)


def test_tracking_metrics_known_error_integral():
    result = tracking_metrics([1, -1, 2, -2], .1)
    assert result["mae_rad_s"] == 1.5
    assert result["iae_rad"] == pytest.approx(.6)
    assert result["rmse_rad_s"] == pytest.approx(np.sqrt(2.5))


def test_antiwindup_and_reset():
    controller = PIController(.001, .05, .12)
    for _ in range(1000):
        action = controller.act({"omega": 0}, 1000, .0001)
    assert action[1] == .12
    assert controller.integral == 0
    assert controller.act({"omega": 0}, 0, .0001)[1] == 0
    controller.act({"omega": 0}, 1, .0001)
    assert controller.integral > 0
    controller.reset()
    assert controller.integral == 0


def test_disturbance_applies_exact_signed_acceleration():
    load = DisturbedLoad(dict(a=.01, b=.01, c=0, j_load=.0029))
    load.set_j_rotor(.0001)
    original = load.mechanical_ode(0, [10], .2)
    load.disturbance_nm = .08
    changed = load.mechanical_ode(0, [10], .2)
    assert (changed - original)[0] == pytest.approx(-.08 / .003)


def short_scenario():
    return {"name": "test", "duration_s": .03, "reference": [[0, 0], [.001, 10]],
            "disturbance": [[0, 0], [.015, .08]]}


def test_reference_and_load_event_alignment(config):
    scenario = short_scenario()
    env, _ = make_environment(config, scenario)
    obs, _ = env.reset(seed=0)
    omega_limit = env.unwrapped.physical_system.limits[0]
    try:
        for k in range(12):
            assert obs[1][0] * omega_limit == schedule_value(scenario["reference"], k, config["tau_s"])
            obs, *_ = env.step([0, 0])
            expected = schedule_value(scenario["reference"], k, config["tau_s"])
            assert env.unwrapped.reference_generator.get_reference()[0] * omega_limit == expected
    finally:
        env.close()


def test_repeated_rollout_is_identical_and_trace_serialization_deterministic(config, tmp_path):
    scenario = short_scenario()
    a, sa = evaluate(config, scenario, PIController(.001, .05, .12), 0)
    b, sb = evaluate(config, scenario, PIController(.001, .05, .12), 0)
    assert sa == sb
    assert sa["completed"]
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert a["disturbance_nm"][149] == 0
    assert a["disturbance_nm"][150] == .08
    write_trace(tmp_path / "a.gz", a)
    write_trace(tmp_path / "b.gz", b)
    assert (tmp_path / "a.gz").read_bytes() == (tmp_path / "b.gz").read_bytes()


def test_constraint_trip_stops_without_reset_or_padding(config):
    class Aggressive:
        def reset(self):
            self.resets = getattr(self, "resets", 0) + 1

        def act(self, *args):
            return [0, 1]

    config["action_norm_limit"] = 1
    actor = Aggressive()
    trace, summary = evaluate(config, short_scenario(), actor, 0)
    assert actor.resets == 1
    assert summary["terminated"] and not summary["completed"]
    assert summary["failure"] == "current_constraint"
    assert summary["samples"] < summary["planned_samples"]
    assert summary["constraint_violating_samples"] == 1
    assert trace["current_constraint_ratio"][-1] > 1
    assert any(e["status"] == "not_reached" for e in summary["events"])
    assert all(e.get("tail_mean_error_rad_s") is None for e in summary["events"])


def test_nonfinite_controller_action_is_reported(config):
    class Broken:
        def reset(self):
            pass

        def act(self, *args):
            return [0, np.nan]

    trace, summary = evaluate(config, short_scenario(), Broken(), 0)
    assert not summary["completed"]
    assert summary["samples"] == 0
    assert "finite" in summary["failure"]


def test_schedule_validation_rejects_off_grid_events(config):
    validate(config)
    config["scenarios"][0]["reference"][1][0] = .10001
    with pytest.raises(ValueError, match="control grid"):
        validate(config)
