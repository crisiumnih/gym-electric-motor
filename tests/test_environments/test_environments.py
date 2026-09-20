import pytest


import numpy as np

import gym_electric_motor as gem


control_tasks = ["TC", "SC", "CC"]
action_types = ["Cont", "Finite"]
ac_motors = ["PMSM", "SynRM", "SCIM", "DFIM"]
dc_motors = ["SeriesDc", "ShuntDc", "PermExDc", "ExtExDc"]
versions = ["v0"]


@pytest.mark.parametrize("version", versions)
@pytest.mark.parametrize("motor", ac_motors + dc_motors)
@pytest.mark.parametrize("control_task", control_tasks)
@pytest.mark.parametrize(["action_type", "tau"], zip(action_types, [1e-4, 1e-5, 1e-4]))
def test_tau(motor, control_task, action_type, version, tau):
    env_id = f"{action_type}-{control_task}-{motor}-{version}"
    env = gem.make(env_id)
    assert env.unwrapped.physical_system.tau == tau


@pytest.mark.parametrize("version", versions)
@pytest.mark.parametrize("ac_motor", ac_motors)
@pytest.mark.parametrize(
    ["control_task", "referenced_states"],
    zip(control_tasks, [["torque"], ["omega"], ["i_sd", "i_sq"]]),
)
@pytest.mark.parametrize("action_type", action_types)
def test_referenced_states_ac(
    ac_motor, control_task, action_type, version, referenced_states
):
    env_id = f"{action_type}-{control_task}-{ac_motor}-{version}"
    env = gem.make(env_id)
    assert env.unwrapped.reference_generator.reference_names == referenced_states


# ---------------------------------------------------------------------------
# Brushless DC Motor Environments
# ---------------------------------------------------------------------------

def test_bldc_env_make():
    env = gem.make("Cont-SC-BLDC-v0")
    # continuous dq voltage action by default
    assert env.action_space.shape == (2,)
    # state vector layout identical to the PMSM speed control environment
    pmsm_env = gem.make("Cont-SC-PMSM-v0")
    assert env.unwrapped.physical_system.state_names == pmsm_env.unwrapped.physical_system.state_names
    assert env.unwrapped.physical_system.tau == 1e-4
    assert env.unwrapped.reference_generator.reference_names == ["omega"]
    assert env.unwrapped.physical_system.limits.shape[0] == 14


def test_bldc_env_abc_action_space():
    env = gem.make("Cont-SC-BLDC-v0", control_space="abc")
    assert env.action_space.shape == (3,)
    env.reset()
    terminated = False
    for _ in range(50):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
            terminated = False


def test_bldc_env_reset_step():
    env = gem.make("Cont-SC-BLDC-v0")
    env.reset()
    obs, reward, terminated, truncated, _ = env.step([0.0, 0.05])
    assert len(obs) == 2
    state, reference = obs
    assert np.asarray(state).shape == (14,)
    assert not terminated and not truncated


def test_bldc_env_star_connection():
    # the phase currents always sum to zero (floating neutral, no common-mode current)
    env = gem.make("Cont-SC-BLDC-v0")
    env.reset()
    ps = env.unwrapped.physical_system
    idx = {n: i for i, n in enumerate(ps.state_names)}
    lim = ps.limits
    for _ in range(2000):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    s = np.asarray(obs[0])
    i_sum = sum(s[idx[n]] * lim[idx[n]] for n in ["i_a", "i_b", "i_c"])
    assert abs(i_sum) < 1e-9


def test_bldc_env_q_voltage_spins_up():
    # a constant positive q-axis voltage accelerates the motor (dq interface aligned
    # like the PMSM: torque from q-current)
    env = gem.make("Cont-SC-BLDC-v0")
    env.reset()
    omega_idx = env.unwrapped.physical_system.state_names.index("omega")
    omega = env.unwrapped.physical_system.limits[omega_idx]
    last_omega = 0.0
    for _ in range(2000):
        obs, _, terminated, truncated, _ = env.step([0.0, 0.05])
        if terminated or truncated:
            break
        last_omega = float(np.asarray(obs[0])[omega_idx] * omega)
    assert last_omega > 3.0
    # a pure d-axis voltage produces no mean torque: the motor stays near standstill
    env.reset()
    last_omega_d = 0.0
    for _ in range(2000):
        obs, _, terminated, truncated, _ = env.step([0.05, 0.0])
        if terminated or truncated:
            break
        last_omega_d = float(np.asarray(obs[0])[omega_idx] * omega)
    assert abs(last_omega_d) < 1.0


def test_bldc_env_dq_observations_use_returned_epsilon():
    env = gem.make("Cont-SC-BLDC-v0")
    env.reset()
    ps = env.unwrapped.physical_system
    idx = {n: i for i, n in enumerate(ps.state_names)}

    for _ in range(250):
        obs, _, terminated, truncated, _ = env.step([0.0, 0.05])
        if terminated or truncated:
            pytest.fail("BLDC terminated before the dq observation consistency check")

    normalized_state = np.asarray(obs[0])
    state = normalized_state * ps.limits
    epsilon = state[idx["epsilon"]]
    i_abc = state[[idx["i_a"], idx["i_b"], idx["i_c"]]]
    u_abc = state[[idx["u_a"], idx["u_b"], idx["u_c"]]]

    np.testing.assert_allclose(
        state[[idx["i_sd"], idx["i_sq"]]],
        ps.abc_to_dq_space(i_abc, epsilon),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        state[[idx["u_sd"], idx["u_sq"]]],
        ps.abc_to_dq_space(u_abc, epsilon),
        atol=1e-12,
    )
