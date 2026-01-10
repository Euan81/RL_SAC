from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def load_reference_npz(ref_npz: str | Path, keys: Sequence[str]) -> np.ndarray:
    """Load reference joint angles from an .npz.

    The file must contain arrays under `keys`, each shaped (P,).
    Returns: (P, J) array [rad].
    """
    ref_npz = Path(ref_npz)
    if not ref_npz.exists():
        raise FileNotFoundError(f"Reference npz not found: {ref_npz}")
    d = np.load(ref_npz, allow_pickle=True)
    q = np.stack([np.asarray(d[k], dtype=np.float64) for k in keys], axis=1)
    # NOTE:
    # The default reference shipped with this project (gait_ref_6d_mujoco.npz)
    # is already expressed in MuJoCo Walker2d joint sign conventions.
    #
    # If you have a reference in a different sign convention, do NOT hard-code
    # per-joint flips here (it silently breaks tracking). Instead, either:
    #   1) bake the flips into a new .npz using tools/make_signed_reference.py, or
    #   2) set TrainCfg.joint_signs to flip the *measured* joint angles.
    return q


def resample_reference(qdes: np.ndarray, P: int) -> np.ndarray:
    """Linear resample along time axis from (P_raw, J) -> (P, J)."""
    qdes = np.asarray(qdes, dtype=np.float64)
    P_raw, J = qdes.shape
    if P_raw < 2:
        raise ValueError(f"Need at least 2 points to resample; got {P_raw}")
    x_old = np.linspace(0.0, 1.0, P_raw)
    x_new = np.linspace(0.0, 1.0, int(P))
    out = np.empty((int(P), J), dtype=np.float64)
    for j in range(J):
        out[:, j] = np.interp(x_new, x_old, qdes[:, j])
    return out
