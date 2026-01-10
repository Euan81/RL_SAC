#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a new gait reference npz with per-joint sign flips.")
    ap.add_argument("--in-npz", type=str, required=True)
    ap.add_argument("--out-npz", type=str, required=True)
    ap.add_argument("--keys", nargs="+", default=["hip_R","knee_R","ankle_R","hip_L","knee_L","ankle_L"])
    ap.add_argument("--signs", nargs="+", type=float, default=[-1, -1, 1, -1, -1, 1])
    args = ap.parse_args()

    in_path = Path(args.in_npz)
    out_path = Path(args.out_npz)
    keys: List[str] = list(args.keys)
    signs = np.asarray(args.signs, dtype=np.float64)

    if signs.shape != (len(keys),):
        raise ValueError(f"--signs must have exactly {len(keys)} values to match --keys")

    data = np.load(in_path, allow_pickle=True)
    out_dict = {}

    for k, s in zip(keys, signs):
        if k not in data:
            raise KeyError(f"Key '{k}' not in {in_path}. Available: {list(data.keys())}")
        out_dict[k] = s * np.asarray(data[k], dtype=np.float64)

    # copy any extras
    for k in data.files:
        if k not in out_dict:
            out_dict[k] = data[k]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out_dict)
    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
