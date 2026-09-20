"""Shared deterministic plant/scenarios. GEM library internals remain unchanged."""

import numpy as np
from gymnasium.spaces import Box

import gym_electric_motor as gem
from gym_electric_motor.core import ReferenceGenerator
from gym_electric_motor.physical_systems import PolynomialStaticLoad, ScipyOdeSolver


def schedule_value(schedule, step, dt):
    """Right-continuous zero-order hold on the integer control grid."""
    values = [value for time, value in schedule if round(time / dt) <= step]
    return values[-1]


class ScheduledReference(ReferenceGenerator):
    def __init__(self, schedule, dt):
        super().__init__()
        self.schedule, self.dt = schedule, dt
        self._reference_names = ["omega"]
        self.reference_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float64)

    def set_modules(self, physical_system):
        super().set_modules(physical_system)
        self.index = physical_system.state_names.index("omega")
        self.limit = physical_system.limits[self.index]
        self._referenced_states = np.zeros(len(physical_system.state_names), dtype=bool)
        self._referenced_states[self.index] = True

    def value(self, step):
        return schedule_value(self.schedule, step, self.dt) / self.limit

    def get_reference(self, *args, **kwargs):
        # step() asks after integration: score against the command just applied.
        result = np.zeros(len(self._referenced_states))
        result[self.index] = self.value(max(0, self._physical_system.k - 1))
        return result

    def get_reference_observation(self, *args, **kwargs):
        # Returned observation carries the command for the next action.
        return np.array([self.value(self._physical_system.k)])


class DisturbedLoad(PolynomialStaticLoad):
    """Add signed shaft torque held constant for each control interval.

    Positive disturbance subtracts from motor torque; it is not sign-dependent friction.
    """

    def __init__(self, parameters):
        super().__init__(load_parameter=parameters)
        self.disturbance_nm = 0.0

    def mechanical_ode(self, t, mechanical_state, torque):
        return super().mechanical_ode(t, mechanical_state, torque - self.disturbance_nm)


class CheckedSolver(ScipyOdeSolver):
    def integrate(self, t):
        state = super().integrate(t)
        if not self._ode.successful() or not np.isfinite(state).all():
            raise RuntimeError("ODE integration failed or returned nonfinite states")
        return state


def make_environment(config, scenario, load_override=None):
    load = load_override if load_override is not None else DisturbedLoad(config["load"])
    env = gem.make(
        "Cont-SC-BLDC-v0",
        motor=dict(motor_parameter=config["motor"]),
        load=load,
        supply=dict(u_nominal=config["supply_v"]),
        converter=dict(interlocking_time=0.0),
        ode_solver=CheckedSolver(**config["solver"]),
        reference_generator=ScheduledReference(scenario["reference"], config["tau_s"]),
        visualization=(),
        tau=config["tau_s"],
        control_space="dq",
    )
    return env, load
