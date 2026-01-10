from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import gymnasium as gym

# Walker2d-v5 joint-name aliases.
# Gymnasium/MuJoCo uses: thigh_joint, leg_joint, foot_joint (+ *_left_joint).
# Our tracking pipeline uses human-friendly names: hip_*, knee_*, ankle_*.
_WALKER2D_ALIASES = {
    "hip_R": "thigh_joint",
    "knee_R": "leg_joint",
    "ankle_R": "foot_joint",
    "hip_L": "thigh_left_joint",
    "knee_L": "leg_left_joint",
    "ankle_L": "foot_left_joint",
}


def _resolve_joint_name(model, name: str) -> str:
    """Return a model joint name that exists, applying Walker2d aliases if needed."""
    try:
        model.joint(name)  # raises KeyError if missing
        return name
    except Exception:
        aliased = _WALKER2D_ALIASES.get(name, name)
        # If alias is the same, let the caller handle the eventual error.
        try:
            model.joint(aliased)
            return aliased
        except Exception:
            return name


def get_joint_angles(env: gym.Env, joint_names: Sequence[str]) -> np.ndarray:
    """Read hinge joint angles [rad] for provided joint names (supports Walker2d aliases)."""
    base = env.unwrapped
    model = getattr(base, "model", None)
    data = getattr(base, "data", None)
    if model is None or data is None:
        raise RuntimeError("Env does not look like a MuJoCo env (missing model/data).")

    q = np.zeros(len(joint_names), dtype=np.float64)
    for i, name in enumerate(joint_names):
        real = _resolve_joint_name(model, name)
        # Try the fast path: data.joint(...).qpos
        try:
            q[i] = float(data.joint(real).qpos[0])
            continue
        except Exception:
            pass

        # Fallback: use qpos address
        try:
            j_id = int(model.joint(real).id)
            adr = int(model.jnt_qposadr[j_id])
            q[i] = float(data.qpos[adr])
        except Exception as e:
            raise KeyError(
                f"Invalid joint name '{name}' (resolved to '{real}'). "
                f"Available joints: {[model.joint(j).name for j in range(model.njnt)]}"
            ) from e
    return q


def get_joint_limits(env: gym.Env, joint_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Read joint limits [rad] for provided joint names (supports Walker2d aliases)."""
    base = env.unwrapped
    model = getattr(base, "model", None)
    if model is None:
        raise RuntimeError("Env does not expose MuJoCo model.")

    qmin = np.full(len(joint_names), -np.pi, dtype=np.float64)
    qmax = np.full(len(joint_names), np.pi, dtype=np.float64)

    for i, name in enumerate(joint_names):
        real = _resolve_joint_name(model, name)
        try:
            rng = np.array(model.joint(real).range, dtype=np.float64)
            if rng.shape == (2,) and np.isfinite(rng).all() and (rng[1] > rng[0]):
                qmin[i], qmax[i] = float(rng[0]), float(rng[1])
        except Exception:
            # unlimited or missing range -> keep wide limits
            pass
    return qmin, qmax
