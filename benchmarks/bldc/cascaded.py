"""Cascaded speed/current PI, in SI units until the final GEM action conversion.

Tuning equations and limitations: docs/bldc_cascaded_pi.md.
The feedforward uses only the fundamental back-EMF, not the exact trapezoidal plant map.
"""

import numpy as np


def derived_gains(config, design):
    motor = config["motor"]
    wc = 2 * np.pi * design["current_bandwidth_hz"]
    wn = 2 * np.pi * design["speed_natural_frequency_hz"]
    inertia = motor["j_rotor"] + config["load"]["j_load"]
    emf_fundamental = 12 / np.pi**2 * motor["k_e"]
    torque_per_iq = 1.5 * emf_fundamental
    return {
        "current_kp_v_per_a": motor["l_s"] * wc,
        "current_ki_v_per_as": motor["r_s"] * wc,
        "current_aw_per_s": wc,
        "speed_kp_a_per_rad_s": (2 * design["speed_damping"] * wn * inertia - config["load"]["b"]) / torque_per_iq,
        "speed_ki_a_per_rad": inertia * wn**2 / torque_per_iq,
        "emf_fundamental_vs_per_rad": emf_fundamental,
        "torque_per_iq_nm_per_a": torque_per_iq,
    }


class DQCurrentController:
    """Two PI axes, fundamental EMF/cross-coupling compensation and vector anti-windup."""

    def __init__(self, config, design):
        self.motor = dict(config["motor"])
        self.voltage_scale = config["supply_v"] / 2
        self.voltage_limit = config["action_norm_limit"] * self.voltage_scale
        self.current_limit = design["current_reference_limit_a"]
        self.gains = derived_gains(config, design)
        self.reset()

    def reset(self):
        self.integral = np.zeros(2)
        self.diagnostics = {}

    def act_current(self, state, reference, dt):
        reference = np.asarray(reference, dtype=float)
        norm = np.linalg.norm(reference)
        target = reference * min(1.0, self.current_limit / max(norm, 1e-30))
        current = np.array([state["i_sd"], state["i_sq"]])
        error = target - current
        electrical_speed = self.motor["p"] * state["omega"]
        feedforward = np.array([
            -electrical_speed * self.motor["l_s"] * current[1],
            electrical_speed * self.motor["l_s"] * current[0]
            + self.gains["emf_fundamental_vs_per_rad"] * state["omega"],
        ])
        # Positional PI: use existing integrator for this action, then update it.
        raw = self.gains["current_kp_v_per_a"] * error + self.integral + feedforward
        raw_norm = np.linalg.norm(raw)
        applied = raw * min(1.0, self.voltage_limit / max(raw_norm, 1e-30))
        self.integral += dt * (
            self.gains["current_ki_v_per_as"] * error
            + self.gains["current_aw_per_s"] * (applied - raw)
        )
        self.diagnostics = {
            "i_d_reference_a": float(target[0]), "i_q_reference_a": float(target[1]),
            "i_d_before_a": float(current[0]), "i_q_before_a": float(current[1]),
            "current_reference_limited": float(norm > self.current_limit + 1e-12),
            "voltage_saturated": float(raw_norm > self.voltage_limit + 1e-12),
            "unsaturated_u_d_v": float(raw[0]), "unsaturated_u_q_v": float(raw[1]),
            "current_integral_d_v": float(self.integral[0]), "current_integral_q_v": float(self.integral[1]),
        }
        return applied / self.voltage_scale


class CascadedPIController:
    """2-DOF speed PI requests iq; id*=0. Cascaded saturation freezes outward integration."""

    def __init__(self, config, design):
        self.current = DQCurrentController(config, design)
        self.gains = self.current.gains
        self.beta = design["speed_setpoint_weight"]
        self.current_limit = design["current_reference_limit_a"]
        self.reset()

    def reset(self):
        self.speed_integral = 0.0
        self.current.reset()
        self.diagnostics = {}

    def act(self, state_si, reference_rad_s, dt):
        error = reference_rad_s - state_si["omega"]
        proportional = self.gains["speed_kp_a_per_rad_s"] * (self.beta * reference_rad_s - state_si["omega"])
        raw_iq = proportional + self.speed_integral
        iq = float(np.clip(raw_iq, -self.current_limit, self.current_limit))
        action = self.current.act_current(state_si, [0.0, iq], dt)
        increment = self.gains["speed_ki_a_per_rad"] * error * dt
        reference_pushes_out = (raw_iq >= self.current_limit and increment > 0) or (raw_iq <= -self.current_limit and increment < 0)
        voltage_pushes_out = bool(self.current.diagnostics["voltage_saturated"]) and error * (iq - state_si["i_sq"]) > 0
        frozen = reference_pushes_out or voltage_pushes_out
        if not frozen:
            self.speed_integral += increment
        self.diagnostics = dict(self.current.diagnostics)
        self.diagnostics.update(
            speed_integral_a=float(self.speed_integral),
            speed_integrator_frozen=float(frozen),
            speed_current_request_a=float(raw_iq),
            current_reference_limited=float(abs(raw_iq) > self.current_limit + 1e-12),
        )
        return action


def make_controller(config):
    if config["controller"]["name"] == "pi_initial":
        from .control import PIController
        return PIController(config["controller"]["kp"], config["controller"]["ki_per_s"], config["action_norm_limit"])
    if config["controller"]["name"] == "pi_cascaded":
        return CascadedPIController(config, config["controller"])
    raise ValueError("Unknown controller name")
