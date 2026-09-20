"""SI speed error, normalized dq voltage output; conditional integration anti-windup."""

import numpy as np


class PIController:
    """Initial PI baseline, not an optimally tuned or cascaded current/speed controller.

    Interface for future controllers: reset(), act(state_si, reference_rad_s, dt).
    state_si is a name-to-SI-value mapping. The evaluator supplies no future reference.
    """

    def __init__(self, kp, ki_per_s, action_limit):
        self.kp, self.ki, self.limit = kp, ki_per_s, action_limit
        self.reset()

    def reset(self):
        self.integral = 0.0

    def act(self, state_si, reference_rad_s, dt):
        error = reference_rad_s - state_si["omega"]
        increment = self.ki * dt * error
        candidate = self.integral + increment
        raw = self.kp * error + candidate
        # Freeze only when the update pushes farther into output saturation.
        if not ((raw > self.limit and increment > 0) or (raw < -self.limit and increment < 0)):
            self.integral = candidate
        return np.array([0.0, np.clip(self.kp * error + self.integral, -self.limit, self.limit)])
