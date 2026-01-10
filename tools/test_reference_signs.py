"""Quick sanity-check for the reference gait sign conventions.

Why this exists
--------------
If your reference is built from another simulator / dataset, it's very common for
one or more joints to have the opposite sign convention to MuJoCo Walker2d.
That will *silently* break tracking rewards (the policy learns to fall early to
avoid accumulating tracking error).

This script does a simple *symmetry* check between right/left legs:
for each joint type (hip/knee/ankle), it compares the right trace against the
left trace shifted by 50% of the gait cycle. If the negated right trace matches
better, we recommend flipping the sign for that joint.

Run:
  uv run python tools/test_reference_signs.py --ref references/gait_ref_6d_mujoco.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PAIR_IDX = {
    "hip": ("hip_R", "hip_L"),
    "knee": ("knee_R", "knee_L"),
    "ankle": ("ankle_R", "ankle_L"),
}


def _load_ref(ref_path: Path) -> dict[str, np.ndarray]:
    d = np.load(ref_path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in d.files:
        out[k] = np.asarray(d[k], dtype=np.float64).reshape(-1)
    return out


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    T = min(a.size, b.size)
    if T == 0:
        return float("nan")
    return float(np.mean((a[:T] - b[:T]) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=str, default="references/gait_ref_6d_mujoco.npz")
    ap.add_argument("--phase-shift", type=float, default=0.5, help="Left shift as fraction of cycle")
    args = ap.parse_args()

    ref_path = Path(args.ref)
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)

    ref = _load_ref(ref_path)
    keys = set(ref.keys())
    print("Available keys:", sorted(keys))

    frac = float(args.phase_shift)
    if not (0.0 <= frac <= 1.0):
        raise ValueError("--phase-shift must be in [0,1]")

    # Determine cycle length from hip_R (fallback: any key)
    any_key = "hip_R" if "hip_R" in ref else sorted(keys)[0]
    P = int(ref[any_key].size)
    shift = int(round(frac * P))

    print(f"Cycle length P={P}, left shift={shift} samples ({frac*100:.0f}%)")
    print()

    suggested = {
        "hip_R": 1,
        "knee_R": 1,
        "ankle_R": 1,
        "hip_L": 1,
        "knee_L": 1,
        "ankle_L": 1,
    }

    for name, (r_key, l_key) in PAIR_IDX.items():
        if r_key not in ref or l_key not in ref:
            print(f"[skip] missing {r_key}/{l_key}")
            continue
        r = ref[r_key]
        l = np.roll(ref[l_key], -shift)  # align left to right

        mse_pos = _mse(r, l)
        mse_neg = _mse(-r, l)
        sign = 1 if mse_pos <= mse_neg else -1
        suggested[r_key] = sign

        pick = "+" if sign == 1 else "-"
        print(f"{name:5s}: mse(r,l)={mse_pos:10.6f}  mse(-r,l)={mse_neg:10.6f}  => recommend {pick}{r_key}")

    print()
    print("Suggested multipliers to apply to the *RIGHT* leg reference:")
    for k in ["hip_R", "knee_R", "ankle_R"]:
        s = suggested[k]
        print(f"  {k}: {s:+d}")

    print()
    print("If you need to permanently bake these into a new .npz, use:")
    print("  uv run python tools/make_signed_reference.py --in-npz <ref.npz> --out-npz <signed.npz> --signs ", end="")
    print(" ".join(str(suggested[k]) for k in ["hip_R","knee_R","ankle_R","hip_L","knee_L","ankle_L"]))


if __name__ == "__main__":
    main()
