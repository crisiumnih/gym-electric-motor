"""
Example: Closed-loop speed control of the brushless DC motor (BLDC) in gym-electric-motor.

The environment 'Cont-SC-BLDC-v0' simulates the T-Motor Antigravity KV100 (36N42P) with a
trapezoidal back-EMF (Pillay & Krishnan 1989) in the phase-variable (abc) frame. The action is a
continuous dq voltage command (2-dim), transformed to abc internally.

This example runs a simple proportional speed controller on the q-axis voltage (u_q = Kp * e_omega)
to demonstrate the environment. The reward and reference generator are the GEM defaults.

Usage:
    python cont_sc_bldc_example.py            # headless run, prints metrics
    python cont_sc_bldc_example.py --plot     # interactive MotorDashboard
"""
import argparse

import numpy as np

import gym_electric_motor as gem
from gym_electric_motor.envs.motors import ActionType, ControlType, Motor, MotorType


def main(plot: bool = False) -> None:
    motor = Motor(MotorType.BrushlessDCMotor, ControlType.SpeedControl, ActionType.Continuous)
    assert motor.env_id() == "Cont-SC-BLDC-v0"

    env = gem.make(
        motor.env_id(),
        motor=dict(motor_parameter=dict(j_rotor=3e-3)),  # rotor + propeller-class load inertia
    )
    physical_system = env.unwrapped.physical_system
    state_names = physical_system.state_names
    omega_idx = state_names.index("omega")
    omega_limit = physical_system.limits[omega_idx]

    # simple clamped PI speed controller on the q-axis voltage.
    # The q-voltage is clamped so that the steady-state current stays below the
    # motor's 40 A rating (u_q <= r_s * i_lim = 3.4 V -> 0.15 normalized).
    omega_ref = 10.0  # rad/s (reachable within the voltage clamp)
    u_q_max = 0.12
    kp = 1e-3   # normalized q-voltage per rad/s of speed error
    ki = 5e-6   # integral gain per control step (dt = 1e-4 s -> ki,cont = 0.05 s^-1)
    n_steps = 30000
    terminated = True
    omega_log = np.zeros(n_steps)
    integral = 0.0
    for k in range(n_steps):
        if terminated:
            state, _ = env.reset()
            terminated = False
        omega = float(np.asarray(state[0])[omega_idx] * omega_limit)
        error = omega_ref - omega
        integral = np.clip(integral + ki * error, -u_q_max, u_q_max)
        u_q = np.clip(kp * error + integral, -u_q_max, u_q_max)
        state, _, terminated, truncated, _ = env.step([0.0, u_q])
        omega_log[k] = omega
        if truncated:
            terminated = True

    omega_ss = omega_log[-2000:]
    print(f"Reference speed:    {omega_ref:.0f} rad/s")
    print(f"Steady-state speed: {omega_ss.mean():.1f} rad/s (min {omega_ss.min():.1f}, max {omega_ss.max():.1f})")
    print(f"Steady-state error: {omega_ref - omega_ss.mean():.1f} rad/s")
    rising = np.argmax(omega_log > 0.9 * omega_ref)
    print(f"Rise time (10-90%): {rising * 1e-4:.2f} s")
    if not plot:
        print("Run with --plot to see the live dashboard.")
    else:
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="show the MotorDashboard")
    args = parser.parse_args()
    main(plot=args.plot)
