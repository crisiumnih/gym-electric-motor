"""GEM adapters dispatched by saved outer-controller contract (plan sect.1).

- v1/v2 7-obs scalar: reuse the frozen `OuterEnv` / `FixV1Env` classes unchanged.
- v3 IASA 19-obs: `IASAEnv` below (new file; frozen sources untouched).
- `env_for_contract`: single dispatch used by validation and re-evaluation so
  policies are always scored under their own contract. Records contract hash
  in the caller protocol, not here.
"""
import copy
import os
import sys
from pathlib import Path
import numpy as np

def _vroom_root():
    env = os.environ.get("VROOM_PATH")
    if env and Path(env).is_dir():
        return env
    cand = "/home/sra/prajwal/vroom"
    if Path(cand, "rl", "outer", "contract.py").exists():
        return cand
    raise ImportError("Vroom shared contract not found; set VROOM_PATH")

VROOM_ROOT = _vroom_root()
if VROOM_ROOT not in sys.path:
    sys.path.insert(0, VROOM_ROOT)

from benchmarks.bldc.environment import schedule_value
from tools.outer_rl.environment import OuterEnv
from tools.outer_fix_v1 import FixV1Env

V1 = 'bldc-outer-speed-v1'
V2 = 'bldc-outer-speed-v2'
V3 = 'bldc-outer-speed-v3'

FAILURE_PENALTY = -5200.0


class IASAEnv(OuterEnv):
    """19-observation two-output IASA GEM adapter (plan sect.2).

    Step = OuterEnv.step verbatim except: (a) 2-dim action through the shared
    IASA arithmetic (direct + integral accumulator + filter, v3 contract);
    (b) 19-dim observation from the shared controller; (c) L1 absolute
    tracking adjustment (AbsoluteTrackingEnv lines); (d) -5200 failure
    override (HorizonEnv line); (e) memory term omitted (no z in v3).
    Timing, limits, trip, history/reset semantics per contract.
    """
    metadata = {'render_modes': []}

    def __init__(self, plant, inner_study, inner_model, scenarios, config=None):
        from gymnasium.spaces import Box
        from rl.outer import controller as ctl
        super().__init__(plant, inner_study, inner_model, scenarios, config=config)
        self.action_space = Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = Box(-1, 1, (19,), dtype=np.float32)
        self._ctl = ctl
        self.cstate = ctl.init_state()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.cstate = self._ctl.init_state()
        s, r = self.state_reference()
        self._ctl.record_history(self.cstate, float(s['omega']), float(r), 0.0)
        return self._obs_iasa(), info

    def _obs_iasa(self):
        s, r = self.state_reference()
        return self._ctl.observe(
            self.cstate, float(s['omega']), float(r), float(s['i_sd']), float(s['i_sq']),
            float(np.sin(s['epsilon'])), float(np.cos(s['epsilon'])))

    def step(self, action):
        if self.done:
            raise RuntimeError('reset required after episode completion')
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError('Two finite actor outputs required')
        self.last_rows = []
        pre, reference = self.state_reference()
        command, _ = self._ctl.apply_action(self.cstate, a[0], a[1])
        previous_raw = self.previous_command
        delta = abs(command - previous_raw) / 3.
        rows = []
        terms = []
        terminated = False
        truncated = False
        failure = None
        duration = round(self.case['duration_s'] / self.tau_i)
        for _ in range(min(self.hold_steps, duration - self.physics_steps)):
            before, ref = self.state_reference()
            self.load.disturbance_nm = schedule_value(self.case['disturbance'], self.physics_steps, self.tau_i)
            voltage = np.asarray(self.current.act_current(before, [0., command], self.tau_i), dtype=float)
            if voltage.shape != (2,) or not np.isfinite(voltage).all():
                raise RuntimeError('Invalid inner action')
            self.observation, _, native_term, native_trunc, _ = self.env.step(voltage)
            after, _ = self.state_reference()
            if not np.isfinite(list(after.values())).all():
                raise RuntimeError('Nonfinite physical state')
            self.physics_steps += 1
            phase_peak = max(abs(after[k]) for k in ['i_a', 'i_b', 'i_c'])
            terminated = bool(native_term or phase_peak > 4.)
            truncated = bool(native_trunc and not terminated)
            failure = ('phase_current_trip' if phase_peak > 4 else 'native_constraint') if terminated else ('native_truncation' if truncated else None)
            components = dict(speed=4. * float(np.clip((ref - after['omega']) / 25, -1, 1)) ** 2,
                current=.5 * float(np.clip(np.hypot(after['i_sd'], after['i_sq']) / 4, 0, 1)) ** 2,
                command_delta=.1 * float(np.clip(delta, 0, 1)) ** 2,
                torque_delta=.1 * float(np.clip(abs(after['torque'] - before['torque']) / .1, 0, 1)) ** 2)
            terms.append(components)
            indices = [self.names.index(n) for n in ['i_a', 'i_b', 'i_c']]
            row = dict(after, **self.current.diagnostics, time_s=self.physics_steps * self.tau_i,
                reference_rad_s=ref, omega_before=float(before['omega']), disturbance_nm=float(self.load.disturbance_nm),
                requested_d=float(voltage[0]), requested_q=float(voltage[1]), action_d=float(voltage[0]), action_q=float(voltage[1]),
                action_clipped=0., action_at_limit=float(np.linalg.norm(voltage) >= self.plant['action_norm_limit'] - 1e-12),
                outer_action_p=float(a[0]), outer_action_i=float(a[1]), outer_command_a=command,
                outer_action_clipped=float(abs(a[0]) > 1 or abs(a[1]) > 1),
                reward=-sum(components.values()), phase_current_ratio=phase_peak / 4., current_constraint_ratio=float(np.sum(np.asarray(self.observation[0])[indices] ** 2)),
                terminated=float(terminated), truncated=float(truncated))
            rows.append(row)
            self.last_rows = rows
            if terminated or truncated:
                break
        if not rows:
            raise RuntimeError('No inner samples in outer transition')
        self.outer_actions += 1
        self._ctl.record_history(self.cstate, float(after['omega']), float(ref), command)
        self.previous_command = command
        truncated = bool(truncated or (not terminated and self.physics_steps >= duration))
        self.done = terminated or truncated
        reward = FAILURE_PENALTY if terminated else -float(np.mean([sum(c.values()) for c in terms]))
        mean_terms = {k: float(np.mean([c[k] for c in terms])) for k in terms[0]}
        mean_terms['failure_penalty'] = FAILURE_PENALTY if terminated else 0.
        for row in rows:
            en = float(np.clip((row['reference_rad_s'] - row['omega']) / 25., -1., 1.))
            difference = 4. * (abs(en) - en ** 2)
            row['reward'] -= difference
        l1cost = []
        for row in rows:
            en = float(np.clip((row['reference_rad_s'] - row['omega']) / 25., -1., 1.))
            l1cost.append(4. * abs(en))
        mean_terms['speed'] = float(np.mean(l1cost))
        if not terminated:
            reward -= float(np.mean([4. * (abs(float(np.clip((row['reference_rad_s'] - row['omega']) / 25., -1., 1.)))
                                          - float(np.clip((row['reference_rad_s'] - row['omega']) / 25., -1., 1.)) ** 2)
                                     for row in rows]))
        info = dict(inner_steps=len(rows), physics_steps=self.physics_steps, outer_actions=self.outer_actions,
            failure=failure, rows=rows, reward_terms=mean_terms, scenario=self.case['name'],
            time_limit=bool(truncated and not failure and self.physics_steps >= duration))
        return self._obs_iasa(), float(reward), bool(terminated), bool(truncated), info


def env_for_contract(version, plant, inner_study, inner_model, scenarios):
    """Instantiate the GEM adapter matching a saved contract version."""
    if version == V1:
        return OuterEnv(plant, inner_study, inner_model, scenarios)
    if version == V2:
        return FixV1Env(plant, inner_study, inner_model, scenarios)
    if version == V3:
        return IASAEnv(plant, inner_study, inner_model, scenarios)
    raise ValueError(f'Unknown contract {version}')
