"""Independent numerical checks for the cascaded controller and comparison metrics."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gym_electric_motor.physical_systems import BrushlessDCMotor, ConstantSpeedLoad
from benchmarks.bldc.cascaded import CascadedPIController, DQCurrentController, derived_gains
from benchmarks.bldc.evaluate import evaluate
from benchmarks.bldc.metrics import current_quality
from benchmarks.bldc.validate_current_loop import run_current_case


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "benchmarks/bldc/config_cascaded.json").read_text())


def state(omega=0, id=0, iq=0):
    return {"omega": omega, "i_sd": id, "i_sq": iq}


def test_torque_conversion_matches_numerical_power_invariant(config):
    motor = BrushlessDCMotor(config["motor"])
    angle = np.linspace(0, 2 * np.pi, 12000, endpoint=False)
    iq = 2.0
    abc = motor.t_32(np.array([-iq * np.sin(angle), iq * np.cos(angle)]))
    mean_torque = config["motor"]["k_e"] * np.sum(motor.bemf_shape(angle) * abc, axis=0).mean()
    assert mean_torque / iq == pytest.approx(derived_gains(config, config["controller"])["torque_per_iq_nm_per_a"], rel=1e-6)


def test_gain_choice_cancels_rl_pole_and_places_outer_poles(config):
    g = derived_gains(config, config["controller"])
    assert g["current_ki_v_per_as"] / g["current_kp_v_per_a"] == pytest.approx(config["motor"]["r_s"] / config["motor"]["l_s"])
    inertia = config["motor"]["j_rotor"] + config["load"]["j_load"]
    poles = np.roots([inertia, config["load"]["b"] + g["torque_per_iq_nm_per_a"] * g["speed_kp_a_per_rad_s"],
                      g["torque_per_iq_nm_per_a"] * g["speed_ki_a_per_rad"]])
    np.testing.assert_allclose(poles, [-2 * np.pi * 5] * 2, rtol=1e-7)


def test_feedforward_signs_and_normalization(config):
    c = DQCurrentController(config, config["controller"])
    s = state(10, .4, -.3)
    a = c.act_current(s, [.4, -.3], config["tau_s"])
    expected = np.array([-.00005 * 210 * -.3, .00005 * 210 * .4 + 12 / np.pi**2 * .0955 * 10])
    np.testing.assert_allclose(a * 22.2, expected, atol=1e-12)


def test_vector_saturation_and_antiwindup_unwind(config):
    c = DQCurrentController(config, config["controller"])
    for _ in range(2000):
        a = c.act_current(state(), [100, 100], config["tau_s"])
        assert np.linalg.norm(a) <= config["action_norm_limit"] + 1e-12
    assert c.diagnostics["current_reference_limited"]
    assert c.diagnostics["voltage_saturated"]
    assert np.linalg.norm(c.integral) < 10
    # Unreachable request ends: emulate a converged zero-error inner plant.
    for _ in range(100):
        c.act_current(state(), [0, 0], config["tau_s"])
    assert np.isfinite(c.integral).all()
    c.reset()
    np.testing.assert_array_equal(c.act_current(state(), [0, 0], config["tau_s"]), [0, 0])


def test_outer_current_limit_and_cascaded_freeze(config):
    c = CascadedPIController(config, config["controller"])
    c.speed_integral = 100
    old = c.speed_integral
    c.act(state(), 100, config["tau_s"])
    assert c.diagnostics["i_q_reference_a"] == 8
    assert c.diagnostics["speed_integrator_frozen"]
    assert c.speed_integral == old
    # Allow integral movement back out of a saturated request.
    c.act(state(), -100, config["tau_s"])
    assert c.speed_integral < old
    c.reset()
    assert c.speed_integral == 0
    assert np.linalg.norm(c.current.integral) == 0


def test_inner_saturation_freezes_outer_even_before_reference_limit(config):
    c = CascadedPIController(config, config["controller"])
    # Near the voltage ceiling, a 5 A request exceeds available headroom at 20 rad/s.
    c.speed_integral = 5 + c.gains["speed_kp_a_per_rad_s"] * 20
    previous = c.speed_integral
    c.act(state(20), 25, config["tau_s"])
    assert c.diagnostics["voltage_saturated"]
    assert not c.diagnostics["current_reference_limited"]
    assert c.diagnostics["speed_integrator_frozen"]
    assert c.speed_integral == previous


def test_harmonic_metric_known_signal():
    angle = np.linspace(0, 12 * np.pi, 6000, endpoint=False)
    trace = {"epsilon": angle, "i_a": 2 * np.sin(angle) + .6 * np.sin(5 * angle) + .8 * np.cos(7 * angle)}
    assert current_quality(trace, slice(None))["tail_phase_a_harmonic_ratio_pct"] == pytest.approx(50, abs=1e-10)


def test_standstill_inner_loop_tracks_steps(config):
    _, result = run_current_case(config, 0)
    assert result["passed"]
    assert result["q_step"]["settling_or_recovery_s"] < .005


def test_current_fixture_ignores_mutated_upstream_default(config, monkeypatch):
    initializer = deepcopy(ConstantSpeedLoad._default_initializer)
    initializer["states"]["omega"] = 60.0
    monkeypatch.setattr(ConstantSpeedLoad, "_default_initializer", initializer)
    trace, result = run_current_case(config, 0)
    assert result["passed"]
    np.testing.assert_array_equal(trace["omega"], 0)


def test_cascade_short_rollout_finite_and_reset_repeatable(config):
    scenario = {"name": "test", "duration_s": .04, "reference": [[0, 0], [.001, 5]], "disturbance": [[0, 0]]}
    controller = CascadedPIController(config, config["controller"])
    a, sa = evaluate(config, scenario, controller, 0)
    b, sb = evaluate(config, scenario, controller, 0)
    assert sa == sb and sa["completed"]
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert np.max(np.abs(a["i_q_reference_a"])) <= 8
    assert np.max(np.hypot(a["action_d"], a["action_q"])) <= .12 + 1e-12
    assert "dq_current_tracking_rmse_a" in sa
