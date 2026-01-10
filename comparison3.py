"""
comparison2.py

Scientifically defensible comparison of Walker2d gait-tracking runs.

UPDATE (hips/knees/ankles MAE/RMSE + CI)
---------------------------------------
This version adds *joint-group* metrics:
  - Hip MAE/RMSE (deg)
  - Knee MAE/RMSE (deg)
  - Ankle MAE/RMSE (deg)
including 95% bootstrap CIs *when per-episode per-joint metrics are available*.

How it works
------------
1) If `evaluation/episode_joint_stats.csv` exists, it is merged into episode_stats and used to compute
   per-episode group metrics + bootstrap CIs.
   Expected columns can be either:
      - "MAE_deg_hip_R", "RMSE_deg_hip_R", ... (preferred)
      - "hip_R_MAE_deg", "hip_R_RMSE_deg", ... (also supported)
2) If per-episode per-joint columns are not found, the script still writes:
      - per_joint_rmse_mae.csv  (point estimates from metrics.json)
      - joint_group_rmse_mae.csv (point estimates from metrics.json)
   but group CIs will be NaN (because episode-level distributions are missing).

To get CIs for hips/knees/ankles, ensure evaluate_run.py writes per-episode per-joint metrics
to `evaluation/episode_joint_stats.csv`.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Metrics
# -----------------------------

# Joint names expected in metrics.json["per_joint"] and (optionally) episode_joint_stats.csv
JOINT_NAMES = ["hip_R", "knee_R", "ankle_R", "hip_L", "knee_L", "ankle_L"]

# Joint groups requested by the user
JOINT_GROUPS: Dict[str, Tuple[str, str]] = {
    "hip": ("hip_R", "hip_L"),
    "knee": ("knee_R", "knee_L"),
    "ankle": ("ankle_R", "ankle_L"),
}

METRICS = {
    # column_name: (pretty_label, direction)
    "return_env": ("Environment return", "higher"),
    "return_total": ("Total return", "higher"),
    "RMSE_deg": ("Tracking RMSE (deg)", "lower"),
    "MAE_deg": ("Tracking MAE (deg)", "lower"),
    "speed": ("Forward speed (m/s)", "higher"),
    "ctrl_cost": ("Control cost", "lower"),
    "ep_len": ("Episode length (steps)", "higher"),
    "survival": ("Survival (0/1)", "higher"),
    # NEW: joint-group metrics (available if episode_joint_stats.csv provides per-episode joint metrics)
    "RMSE_deg_hip": ("Hip RMSE (deg)", "lower"),
    "MAE_deg_hip": ("Hip MAE (deg)", "lower"),
    "RMSE_deg_knee": ("Knee RMSE (deg)", "lower"),
    "MAE_deg_knee": ("Knee MAE (deg)", "lower"),
    "RMSE_deg_ankle": ("Ankle RMSE (deg)", "lower"),
    "MAE_deg_ankle": ("Ankle MAE (deg)", "lower"),
}


# -----------------------------
# Data model
# -----------------------------


@dataclass
class RunData:
    run_dir: Path
    name: str
    mode: str
    seed: Optional[int]
    metrics_json: Dict[str, Any]
    ep: pd.DataFrame  # episode_stats.csv (possibly merged + augmented)


DEFAULT_ORDER = ["baseline", "paper6", "exp6"]


# -----------------------------
# Basic helpers
# -----------------------------


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def mean(x: np.ndarray) -> float:
    x = _finite(x)
    return float(np.mean(x)) if x.size else float("nan")


def median(x: np.ndarray) -> float:
    x = _finite(x)
    return float(np.median(x)) if x.size else float("nan")


def bootstrap_ci(
    x: np.ndarray,
    stat_fn: Callable[[np.ndarray], float] = mean,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for a 1D sample."""
    x = _finite(x)
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        v = float(x[0])
        return v, v

    boots = np.empty(int(n_boot), dtype=float)
    n = x.size
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boots[i] = stat_fn(x[idx])

    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


# -----------------------------
# Episode per-joint merge + joint-group augmentation
# -----------------------------


def _find_joint_col(df: pd.DataFrame, metric: str, joint: str) -> Optional[str]:
    """
    Find a per-episode per-joint metric column in a forgiving way.

    Supported patterns:
      1) f"{metric}_{joint}"         e.g., "MAE_deg_hip_R"
      2) f"{joint}_{metric}"         e.g., "hip_R_MAE_deg"
    """
    c1 = f"{metric}_{joint}"
    if c1 in df.columns:
        return c1
    c2 = f"{joint}_{metric}"
    if c2 in df.columns:
        return c2
    return None


def maybe_merge_episode_joint_stats(ep: pd.DataFrame, eval_dir: Path) -> pd.DataFrame:
    """
    If evaluation/episode_joint_stats.csv exists, merge it into episode_stats.

    The joint CSV should have one row per episode, in the same order as episode_stats.csv.
    If it contains an 'episode' column, merge on that; otherwise merge by row index.
    """
    joint_path = eval_dir / "episode_joint_stats.csv"
    if not joint_path.exists():
        return ep

    joint_df = pd.read_csv(joint_path)
    if joint_df.empty:
        return ep

    # Merge by an explicit episode id if present; else align by row order
    if "episode" in joint_df.columns:
        ep2 = ep.copy()
        ep2["episode"] = np.arange(len(ep2), dtype=int)
        merged = ep2.merge(joint_df, on="episode", how="left", suffixes=("", "_joint"))
        merged = merged.drop(columns=["episode"])
        return merged

    # Align by index
    joint_df = joint_df.reset_index(drop=True)
    ep_df = ep.reset_index(drop=True)
    # Avoid duplicate column names
    overlap = set(ep_df.columns) & set(joint_df.columns)
    joint_df = joint_df.rename(columns={c: f"{c}_joint" for c in overlap})
    return pd.concat([ep_df, joint_df], axis=1)


def add_joint_group_columns(ep: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-episode group columns:
      MAE_deg_hip, RMSE_deg_hip, etc.

    Requires per-episode per-joint columns for both L/R sides.
    """
    ep2 = ep.copy()

    # For each group, compute per-episode aggregate if both sides available
    for group, (jr, jl) in JOINT_GROUPS.items():
        # MAE: average of left/right MAE per episode
        c_mae_r = _find_joint_col(ep2, "MAE_deg", jr)
        c_mae_l = _find_joint_col(ep2, "MAE_deg", jl)
        if c_mae_r and c_mae_l:
            ep2[f"MAE_deg_{group}"] = 0.5 * (ep2[c_mae_r].astype(float) + ep2[c_mae_l].astype(float))

        # RMSE: combine as sqrt(mean(MSE_R, MSE_L)) per episode
        c_rmse_r = _find_joint_col(ep2, "RMSE_deg", jr)
        c_rmse_l = _find_joint_col(ep2, "RMSE_deg", jl)
        if c_rmse_r and c_rmse_l:
            rr = ep2[c_rmse_r].astype(float).to_numpy()
            ll = ep2[c_rmse_l].astype(float).to_numpy()
            ep2[f"RMSE_deg_{group}"] = np.sqrt(0.5 * (rr * rr + ll * ll))

    return ep2


# -----------------------------
# Loading runs
# -----------------------------


def infer_mode(name: str) -> str:
    lname = name.lower()
    for m in DEFAULT_ORDER:
        if m in lname:
            return m
    return name


def load_run(run_dir: Path, *, episodes: int) -> RunData:
    run_dir = Path(run_dir)
    name = run_dir.name

    # Load config.json for seed (used to decide paired vs unpaired comparisons)
    seed = None
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = _load_json(cfg_path)
        if "seed" in cfg:
            try:
                seed = int(cfg["seed"])  # type: ignore[arg-type]
            except Exception:
                seed = None

    eval_dir = run_dir / "evaluation"
    metrics_path = eval_dir / "metrics.json"
    ep_path = eval_dir / "episode_stats.csv"

    if not metrics_path.exists() or not ep_path.exists():
        # Try to auto-evaluate if possible
        try:
            from evaluate_run import evaluate  # local project import

            evaluate(run_dir, episodes=int(episodes), mode=None)
        except Exception as e:
            raise FileNotFoundError(
                f"Missing evaluation files in {eval_dir}. "
                "Run `python evaluate_run.py --run-dir <RUN> --episodes <N>` first.\n"
                f"Also tried to auto-evaluate but failed: {e}"
            )

    metrics_json = _load_json(metrics_path)
    mode = str(metrics_json.get("mode", infer_mode(name)))

    ep = pd.read_csv(ep_path)

    # NEW: optionally merge per-episode per-joint stats, then compute hip/knee/ankle group columns
    ep = maybe_merge_episode_joint_stats(ep, eval_dir=eval_dir)
    ep = add_joint_group_columns(ep)

    return RunData(run_dir=run_dir, name=name, mode=mode, seed=seed, metrics_json=metrics_json, ep=ep)


# -----------------------------
# Tables
# -----------------------------


def build_summary_table(
    runs: List[RunData],
    *,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for rd in runs:
        for col, (label, direction) in METRICS.items():
            if col not in rd.ep.columns:
                continue
            x = rd.ep[col].to_numpy(dtype=float)
            n = int(np.sum(np.isfinite(x)))
            mu = mean(x)
            lo, hi = bootstrap_ci(x, mean, n_boot=n_boot, alpha=alpha, rng=rng)
            med = median(x)
            rows.append(
                {
                    "run": rd.name,
                    "mode": rd.mode,
                    "metric": col,
                    "label": label,
                    "direction": direction,
                    "n": n,
                    "mean": mu,
                    "ci_low": lo,
                    "ci_high": hi,
                    "median": med,
                }
            )
    return pd.DataFrame(rows)


def write_per_joint_table(runs: List[RunData], out_dir: Path) -> None:
    """Extract per-joint MAE/RMSE from metrics.json (point estimates)."""
    rows: List[Dict[str, object]] = []
    for rd in runs:
        per_joint = rd.metrics_json.get("per_joint", {})
        if not isinstance(per_joint, dict):
            continue
        for jname, vals in per_joint.items():
            if not isinstance(vals, dict):
                continue
            rows.append(
                {
                    "mode": rd.mode,
                    "joint": str(jname),
                    "MAE_deg": float(vals.get("MAE_deg", float("nan"))),
                    "RMSE_deg": float(vals.get("RMSE_deg", float("nan"))),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_dir / "per_joint_rmse_mae.csv", index=False)


def write_joint_group_table(
    runs: List[RunData],
    out_dir: Path,
    *,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> None:
    """
    Writes hips/knees/ankles MAE/RMSE + (bootstrap) CI when episode-level joint metrics exist.
    Falls back to metrics.json point estimates when they don't (CI will be NaN).
    """
    rows: List[Dict[str, object]] = []

    for rd in runs:
        # 1) If per-episode group columns exist, compute mean + bootstrap CI (best case)
        for group in JOINT_GROUPS.keys():
            for metric in ["MAE_deg", "RMSE_deg"]:
                col = f"{metric}_{group}"
                if col in rd.ep.columns:
                    x = rd.ep[col].to_numpy(dtype=float)
                    lo, hi = bootstrap_ci(x, mean, n_boot=n_boot, alpha=alpha, rng=rng)
                    rows.append(
                        {
                            "mode": rd.mode,
                            "group": group,
                            "metric": metric,
                            "mean": mean(x),
                            "ci_low": lo,
                            "ci_high": hi,
                            "n": int(np.sum(np.isfinite(x))),
                            "source": "episode_joint_stats.csv",
                        }
                    )

        # 2) Fallback: compute group point estimates from metrics.json per_joint
        #    (no CI unless episode-level exists)
        per_joint = rd.metrics_json.get("per_joint", {})
        if isinstance(per_joint, dict):
            for group, (jr, jl) in JOINT_GROUPS.items():
                v_r = per_joint.get(jr, {})
                v_l = per_joint.get(jl, {})
                if not (isinstance(v_r, dict) and isinstance(v_l, dict)):
                    continue

                mae_r = float(v_r.get("MAE_deg", float("nan")))
                mae_l = float(v_l.get("MAE_deg", float("nan")))
                rmse_r = float(v_r.get("RMSE_deg", float("nan")))
                rmse_l = float(v_l.get("RMSE_deg", float("nan")))

                # If episode-level rows already exist above, don't duplicate fallback rows
                has_episode_cols = (f"MAE_deg_{group}" in rd.ep.columns) or (f"RMSE_deg_{group}" in rd.ep.columns)
                if has_episode_cols:
                    continue

                # Group MAE: average L/R
                rows.append(
                    {
                        "mode": rd.mode,
                        "group": group,
                        "metric": "MAE_deg",
                        "mean": 0.5 * (mae_r + mae_l),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "n": 0,
                        "source": "metrics.json (no CI)",
                    }
                )
                # Group RMSE: sqrt(mean(MSE_R, MSE_L))
                rows.append(
                    {
                        "mode": rd.mode,
                        "group": group,
                        "metric": "RMSE_deg",
                        "mean": math.sqrt(0.5 * (rmse_r * rmse_r + rmse_l * rmse_l)),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "n": 0,
                        "source": "metrics.json (no CI)",
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        # nice ordering
        df["group"] = pd.Categorical(df["group"], categories=["hip", "knee", "ankle"], ordered=True)
        df["metric"] = pd.Categorical(df["metric"], categories=["MAE_deg", "RMSE_deg"], ordered=True)
        df = df.sort_values(["mode", "group", "metric"]).reset_index(drop=True)
        df.to_csv(out_dir / "joint_group_rmse_mae.csv", index=False)


# -----------------------------
# Plotting (kept minimal here)
# -----------------------------


def plot_rmse_box(runs: List[RunData], out_dir: Path, dpi: int = 600) -> None:
    """Boxplot for overall RMSE_deg (if present)."""
    labels = []
    data = []
    for rd in runs:
        if "RMSE_deg" not in rd.ep.columns:
            continue
        labels.append(rd.mode)
        data.append(rd.ep["RMSE_deg"].to_numpy(dtype=float))

    if not data:
        return

    fig = plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Tracking RMSE (deg)")
    plt.title("Episode RMSE distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_box.png", dpi=int(dpi))
    plt.close(fig)


# -----------------------------
# CLI
# -----------------------------


def _sort_runs(runs: List[RunData]) -> List[RunData]:
    order = {m: i for i, m in enumerate(DEFAULT_ORDER)}
    return sorted(runs, key=lambda r: order.get(r.mode, 999))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        default=["runs/baseline_hipRneg", "runs/paper6_hipRneg", "runs/exp6_hipRneg"],
        help="Run directories (each containing config.json and evaluation/*).",
    )
    ap.add_argument("--out-dir", type=str, default="runs/compare2")
    ap.add_argument("--episodes", type=int, default=20, help="Used only if evaluation files are missing.")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))

    runs = [_sort_runs([load_run(Path(r), episodes=int(args.episodes)) for r in args.runs])][0]

    # Summary table with CIs (includes hip/knee/ankle if available)
    summary = build_summary_table(
        runs, n_boot=int(args.n_boot), alpha=float(args.alpha), rng=rng
    )
    if not summary.empty:
        summary.to_csv(out_dir / "summary_stats.csv", index=False)

    # Existing per-joint point-estimate table + NEW joint-group table
    write_per_joint_table(runs, out_dir)
    write_joint_group_table(runs, out_dir, n_boot=int(args.n_boot), alpha=float(args.alpha), rng=rng)

    # Basic plot(s)
    plot_rmse_box(runs, out_dir, dpi=int(args.dpi))

    # Console quickview (prints any available metrics, including hip/knee/ankle if present)
    print("\n=== Mean + bootstrap CI (if available) ===")
    for metric in [
        "MAE_deg",
        "RMSE_deg",
        "MAE_deg_hip",
        "RMSE_deg_hip",
        "MAE_deg_knee",
        "RMSE_deg_knee",
        "MAE_deg_ankle",
        "RMSE_deg_ankle",
    ]:
        if metric not in METRICS:
            continue
        lab = METRICS[metric][0]
        print(f"\n{lab} [{metric}]")
        any_printed = False
        for rd in runs:
            if metric not in rd.ep.columns:
                continue
            x = rd.ep[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(x, mean, n_boot=int(args.n_boot), alpha=float(args.alpha), rng=rng)
            print(f"  {rd.mode:>8s}: {mean(x): .4f}   (CI {lo: .4f}, {hi: .4f})")
            any_printed = True
        if not any_printed:
            print("  (not available: requires evaluation/episode_joint_stats.csv with per-episode per-joint metrics)")

    print(f"\nWrote results to: {out_dir}")


if __name__ == "__main__":
    main()
