"""Common actuator boundary for classical, learned and exported controllers."""

import numpy as np


def project_voltage(requested, limit):
    """Radial projection of normalized dq voltage onto a disk of radius limit."""
    action = np.asarray(requested, dtype=float)
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError("Controller must return two finite normalized dq voltage commands")
    if not np.isfinite(limit) or not 0 < limit <= 1:
        raise ValueError("Voltage limit must be finite and in (0, 1]")
    norm = np.linalg.norm(action)
    return action * min(1.0, limit / max(norm, 1e-30))
