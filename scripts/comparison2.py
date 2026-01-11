"""comparison2.py

Scientifically defensible comparison of Walker2d gait-tracking runs.

What it does
------------
Given multiple `run-dir`s (each produced by `train_sac_walker2d.py` + `evaluate_run.py`),
this script:

1) Loads per-episode metrics from `RUN_DIR/evaluation/episode_stats.csv`.
2) Computes robust descriptive statistics (mean/median, IQR) + 95% bootstrap CIs.
3) Runs *distribution-free* hypothesis tests using permutation tests:
   - Unpaired permutation test if runs were evaluated with different seeds.
   - Paired sign-flip permutation test if runs share the same evaluation seed.
4) Reports effect sizes (Cohen's d and Cliff's delta).
5) Applies Holm correction across the 3 pairwise comparisons per metric.
6) Saves publication-ready plots and CSV tables.

Usage
-----
    python comparison2.py \
      --runs runs/baseline_hipRneg runs/paper6_hipRneg runs/exp6_hipRneg \
      --out-dir runs/compare2 \
      --episodes 20

Outputs
-------
OUT_DIR/
  summary_stats.csv
  pairwise_comparisons.csv
  per_joint_rmse_mae.csv
  plots/
    metric_<name>.png/.pdf
    diff_<name>.png/.pdf
    speed_vs_rmse.png/.pdf

Notes
-----
• With only one training seed per method, these comparisons quantify how the *trained policies*
  behave over evaluation episodes. They do not capture variability across training runs.
• Permutation tests + bootstrap CIs avoid strong normality assumptions and are appropriate
  for small-n evaluation (e.g., 20 episodes).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "comparison2.py requires pandas (it should already be in requirements.txt).\n"
        f"Import error: {e}"
    )

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "comparison2.py requires matplotlib (it should already be in requirements.txt).\n"
        f"Import error: {e}"
    )


# -----------------------------
# Configuration
# -----------------------------


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
}


DEFAULT_ORDER = ["baseline", "Sang-model", "New model"]
MODE_LABELS = {
    "baseline": "Baseline",
    "paper6": "Sang model",
    "exp6": "New model",
}



# -----------------------------
# Small stats helpers (no SciPy)
# -----------------------------


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def mean(x: np.ndarray) -> float:
    x = _finite(x)
    return float(np.mean(x)) if x.size else float("nan")


def std(x: np.ndarray) -> float:
    x = _finite(x)
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1))


def median(x: np.ndarray) -> float:
    x = _finite(x)
    return float(np.median(x)) if x.size else float("nan")


def iqr(x: np.ndarray) -> float:
    x = _finite(x)
    if x.size == 0:
        return float("nan")
    q25, q75 = np.percentile(x, [25, 75])
    return float(q75 - q25)


def bootstrap_ci(
    x: np.ndarray,
    stat_fn: Callable[[np.ndarray], float] = mean,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Percentile bootstrap CI."""
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


def bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    paired: bool,
    stat_fn: Callable[[np.ndarray], float] = mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Bootstrap CI for (stat(b) - stat(a))."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if paired:
        # Filter to indices where both are finite
        mask = np.isfinite(a) & np.isfinite(b)
        a2, b2 = a[mask], b[mask]
        if a2.size == 0:
            return float("nan"), float("nan")
        if a2.size == 1:
            d = float(b2[0] - a2[0])
            return d, d
        diffs = b2 - a2
        n = diffs.size
        boots = np.empty(int(n_boot), dtype=float)
        for i in range(int(n_boot)):
            idx = rng.integers(0, n, size=n)
            boots[i] = stat_fn(diffs[idx])
        lo = float(np.percentile(boots, 100 * (alpha / 2)))
        hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
        return lo, hi

    # Unpaired
    a2 = _finite(a)
    b2 = _finite(b)
    if a2.size == 0 or b2.size == 0:
        return float("nan"), float("nan")
    if a2.size == 1 and b2.size == 1:
        d = float(stat_fn(b2) - stat_fn(a2))
        return d, d

    boots = np.empty(int(n_boot), dtype=float)
    na, nb = a2.size, b2.size
    for i in range(int(n_boot)):
        ia = rng.integers(0, na, size=na)
        ib = rng.integers(0, nb, size=nb)
        boots[i] = stat_fn(b2[ib]) - stat_fn(a2[ia])
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def permutation_test_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    *,
    paired: bool,
    stat_fn: Callable[[np.ndarray], float] = mean,
    n_perm: int = 20_000,
    rng: np.random.Generator,
) -> float:
    """Two-sided permutation p-value for difference in statistic (stat(b) - stat(a))."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if paired:
        mask = np.isfinite(a) & np.isfinite(b)
        a2, b2 = a[mask], b[mask]
        if a2.size == 0:
            return float("nan")
        diffs = b2 - a2
        obs = float(stat_fn(diffs))
        # sign-flip test
        count = 0
        n = diffs.size
        for _ in range(int(n_perm)):
            signs = rng.choice([-1.0, 1.0], size=n, replace=True)
            val = float(stat_fn(diffs * signs))
            if abs(val) >= abs(obs):
                count += 1
        return float((count + 1) / (n_perm + 1))

    a2 = _finite(a)
    b2 = _finite(b)
    if a2.size == 0 or b2.size == 0:
        return float("nan")
    obs = float(stat_fn(b2) - stat_fn(a2))

    pool = np.concatenate([a2, b2], axis=0)
    n_a = a2.size
    n = pool.size

    count = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(n)
        aa = pool[perm[:n_a]]
        bb = pool[perm[n_a:]]
        val = float(stat_fn(bb) - stat_fn(aa))
        if abs(val) >= abs(obs):
            count += 1
    return float((count + 1) / (n_perm + 1))


def cohens_d(a: np.ndarray, b: np.ndarray, *, paired: bool) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if paired:
        mask = np.isfinite(a) & np.isfinite(b)
        d = b[mask] - a[mask]
        d = _finite(d)
        if d.size < 2:
            return float("nan")
        return float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))

    a2, b2 = _finite(a), _finite(b)
    if a2.size < 2 or b2.size < 2:
        return float("nan")
    ma, mb = float(np.mean(a2)), float(np.mean(b2))
    sa, sb = float(np.std(a2, ddof=1)), float(np.std(b2, ddof=1))
    sp = math.sqrt(((a2.size - 1) * sa * sa + (b2.size - 1) * sb * sb) / (a2.size + b2.size - 2))
    if sp <= 0:
        return float("nan")
    return float((mb - ma) / sp)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta for unpaired samples (distribution-free effect size in [-1, 1])."""
    a2, b2 = _finite(a), _finite(b)
    if a2.size == 0 or b2.size == 0:
        return float("nan")
    # O(n*m) but here n,m <= 20
    gt = 0
    lt = 0
    for x in a2:
        for y in b2:
            if y > x:
                gt += 1
            elif y < x:
                lt += 1
    denom = a2.size * b2.size
    return float((gt - lt) / denom)


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    """Holm–Bonferroni adjustment (controls FWER)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    out = np.full(m, np.nan, dtype=float)
    # Handle NaNs: leave as NaN
    finite_idx = np.where(np.isfinite(p))[0]
    if finite_idx.size == 0:
        return out.tolist()

    pf = p[finite_idx]
    order = np.argsort(pf)
    adj = np.empty_like(pf)
    for i, oi in enumerate(order):
        adj[oi] = min((pf.size - i) * pf[oi], 1.0)
    # enforce monotonicity
    for i in range(1, pf.size):
        prev = adj[order[i - 1]]
        cur = adj[order[i]]
        if cur < prev:
            adj[order[i]] = prev

    out[finite_idx] = adj
    return out.tolist()


# -----------------------------
# IO and run metadata
# -----------------------------


@dataclass
class RunData:
    run_dir: Path
    name: str
    mode: str
    seed: Optional[int]
    metrics_json: Dict[str, object]
    ep: pd.DataFrame


def infer_mode(run_name: str) -> str:
    n = run_name.lower()
    if "baseline" in n:
        return "baseline"
    if "paper" in n:
        return "paper6"
    if "exp" in n:
        return "exp6"
    return run_name


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


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

    # Ensure evaluation exists
    eval_dir = run_dir / "evaluation"
    metrics_path = eval_dir / "metrics.json"
    ep_path = eval_dir / "episode_stats.csv"

    if not metrics_path.exists() or not ep_path.exists():
        # Best-effort: run evaluation if available
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
    return RunData(run_dir=run_dir, name=name, mode=mode, seed=seed, metrics_json=metrics_json, ep=ep)


def ensure_out_dir(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    return out_dir


def savefig(path_no_ext: Path, dpi: int = 600) -> None:
    plt.tight_layout()
    plt.savefig(str(path_no_ext.with_suffix(".png")), dpi=int(dpi))
    plt.savefig(str(path_no_ext.with_suffix(".pdf")))
    plt.close()


# -----------------------------
# Core analysis
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
                    "mean_ci_low": lo,
                    "mean_ci_high": hi,
                    "std": std(x),
                    "median": med,
                    "iqr": iqr(x),
                }
            )
    return pd.DataFrame(rows)


def build_pairwise_table(
    runs: List[RunData],
    *,
    paired: bool,
    n_boot: int,
    n_perm: int,
    alpha: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    # Pre-extract arrays for speed
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for rd in runs:
        data[rd.mode] = {c: rd.ep[c].to_numpy(dtype=float) for c in rd.ep.columns if c in METRICS}

    modes = [rd.mode for rd in runs]

    for metric, (label, direction) in METRICS.items():
        # gather pvals for Holm (3 comparisons per metric)
        pvals_metric: List[float] = []
        row_idxs: List[int] = []

        for a_mode, b_mode in combinations(modes, 2):
            if metric not in data.get(a_mode, {}) or metric not in data.get(b_mode, {}):
                continue
            a = data[a_mode][metric]
            b = data[b_mode][metric]

            # If paired, ensure equal length by truncation (evaluation should be same episodes)
            if paired:
                L = min(len(a), len(b))
                a_use = a[:L]
                b_use = b[:L]
            else:
                a_use, b_use = a, b

            diff = mean(b_use) - mean(a_use)
            ci_lo, ci_hi = bootstrap_diff_ci(
                a_use,
                b_use,
                paired=paired,
                stat_fn=mean,
                n_boot=n_boot,
                alpha=alpha,
                rng=rng,
            )
            p_perm = permutation_test_pvalue(
                a_use,
                b_use,
                paired=paired,
                stat_fn=mean,
                n_perm=n_perm,
                rng=rng,
            )

            d = cohens_d(a_use, b_use, paired=paired)
            cd = cliffs_delta(a_use, b_use) if not paired else float("nan")

            n_a = int(np.sum(np.isfinite(a_use)))
            n_b = int(np.sum(np.isfinite(b_use)))

            rows.append(
                {
                    "metric": metric,
                    "label": label,
                    "direction": direction,
                    "paired": bool(paired),
                    "A": a_mode,
                    "B": b_mode,
                    "n_A": n_a,
                    "n_B": n_b,
                    "mean_A": mean(a_use),
                    "mean_B": mean(b_use),
                    "diff_mean_B_minus_A": diff,
                    "diff_ci_low": ci_lo,
                    "diff_ci_high": ci_hi,
                    "p_perm": p_perm,
                    "cohens_d": d,
                    "cliffs_delta": cd,
                }
            )

            pvals_metric.append(p_perm)
            row_idxs.append(len(rows) - 1)

        # Holm adjust within this metric
        if pvals_metric:
            adj = holm_adjust(pvals_metric)
            for idx, p_adj in zip(row_idxs, adj):
                rows[idx]["p_perm_holm"] = p_adj

    return pd.DataFrame(rows)


def write_per_joint_table(runs: List[RunData], out_dir: Path) -> None:
    """Extract per-joint MAE/RMSE from metrics.json for an easy report table."""
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
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "per_joint_rmse_mae.csv", index=False)


# -----------------------------
# Plotting
# -----------------------------


def plot_metric_distributions(
    runs: List[RunData],
    *,
    out_dir: Path,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
    dpi: int,
) -> None:
    modes = [rd.mode for rd in runs]
    for metric, (label, _direction) in METRICS.items():
        # Collect per-episode samples
        series: List[np.ndarray] = []
        for rd in runs:
            if metric in rd.ep.columns:
                series.append(rd.ep[metric].to_numpy(dtype=float))
            else:
                series.append(np.array([np.nan], dtype=float))

        # Bootstrap CI for each mean
        means = [mean(x) for x in series]
        cis = [bootstrap_ci(x, mean, n_boot=n_boot, alpha=alpha, rng=rng) for x in series]

        plt.figure(figsize=(7.2, 4.6))
        # Boxplot
        plt.boxplot(series, labels=modes, showmeans=False, showfliers=False)
        # Jittered points
        for i, x in enumerate(series, start=1):
            xf = _finite(x)
            if xf.size == 0:
                continue
            jitter = rng.normal(loc=i, scale=0.05, size=xf.size)
            plt.plot(jitter, xf, "o", markersize=3, alpha=0.35)
        # Mean ± bootstrap CI
        xs = np.arange(1, len(modes) + 1)
        yerr_low = [m - lo if np.isfinite(m) and np.isfinite(lo) else 0.0 for m, (lo, hi) in zip(means, cis)]
        yerr_high = [hi - m if np.isfinite(m) and np.isfinite(hi) else 0.0 for m, (lo, hi) in zip(means, cis)]
        plt.errorbar(xs, means, yerr=[yerr_low, yerr_high], fmt="s", capsize=5)

        plt.title(f"{label} (per-episode)\nMean ± 95% bootstrap CI")
        plt.ylabel(label)
        plt.ylim(bottom=0)
        plt.minorticks_on()
        plt.grid(which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.35)
        plt.grid(which="minor", axis="y", linestyle="--", linewidth=0.6, alpha=0.20)

        savefig(out_dir / "plots" / f"metric_{metric}", dpi=dpi)


def plot_pairwise_diffs(
    pairwise_df: pd.DataFrame,
    *,
    out_dir: Path,
    dpi: int,
) -> None:
    """Simple forest plot per metric: diff mean (B-A) with 95% CI for each pair."""
    if pairwise_df.empty:
        return
    for metric, sub in pairwise_df.groupby("metric", sort=False):
        label = str(sub["label"].iloc[0])
        # Sort for deterministic y-order
        sub = sub.sort_values(["A", "B"], ascending=True)
        y = np.arange(len(sub))

        diffs = sub["diff_mean_B_minus_A"].to_numpy(dtype=float)
        lo = sub["diff_ci_low"].to_numpy(dtype=float)
        hi = sub["diff_ci_high"].to_numpy(dtype=float)

        yerr_low = diffs - lo
        yerr_high = hi - diffs
        labels = [f"{a} → {b}" for a, b in zip(sub["A"].tolist(), sub["B"].tolist())]

        plt.figure(figsize=(7.2, 3.8))
        plt.axvline(0.0, linewidth=1)
        plt.errorbar(diffs, y, xerr=[yerr_low, yerr_high], fmt="o", capsize=5)
        plt.yticks(y, labels)
        plt.ylim(bottom=0)
        plt.minorticks_on()
        plt.grid(which="major", axis="x", linestyle="-", linewidth=0.8, alpha=0.35)
        plt.grid(which="minor", axis="x", linestyle="--", linewidth=0.6, alpha=0.20)

        plt.xlabel("Difference in mean (B − A)")
        plt.title(f"{label}: pairwise differences (95% bootstrap CI)")
        savefig(out_dir / "plots" / f"diff_{metric}", dpi=dpi)


def plot_speed_vs_rmse(runs: List[RunData], *, out_dir: Path, dpi: int) -> None:
    plt.figure(figsize=(7.2, 4.6))
    for rd in runs:
        if "speed" not in rd.ep.columns or "RMSE_deg" not in rd.ep.columns:
            continue
        sp = rd.ep["speed"].to_numpy(dtype=float)
        rm = rd.ep["RMSE_deg"].to_numpy(dtype=float)
        mask = np.isfinite(sp) & np.isfinite(rm)
        plt.plot(sp[mask], rm[mask], "o", markersize=4, alpha=0.6, label=MODE_LABELS.get(rd.mode, rd.mode))

    plt.xlabel("Forward speed (m/s)")
    plt.ylabel("Tracking RMSE (deg)")
    plt.ylim(bottom=0)
    plt.legend()
    plt.minorticks_on()
    plt.grid(which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.35)
    plt.grid(which="minor", axis="y", linestyle="--", linewidth=0.6, alpha=0.20)

    savefig(out_dir / "plots" / "speed_vs_rmse", dpi=dpi)


# -----------------------------
# Main
# -----------------------------


def _sort_runs(runs: List[RunData]) -> List[RunData]:
    order = {m: i for i, m in enumerate(DEFAULT_ORDER)}
    return sorted(runs, key=lambda r: order.get(r.mode, 999))


def decide_paired(runs: List[RunData], forced: Optional[bool]) -> bool:
    if forced is not None:
        return bool(forced)
    seeds = [r.seed for r in runs]
    if all(s is not None for s in seeds) and len(set(seeds)) == 1:
        return True
    return False


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
    ap.add_argument("--paired", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-perm", type=int, default=20_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    out_dir = ensure_out_dir(Path(args.out_dir))

    runs = [load_run(Path(r), episodes=int(args.episodes)) for r in args.runs]
    runs = _sort_runs(runs)

    forced: Optional[bool]
    if args.paired == "auto":
        forced = None
    elif args.paired == "true":
        forced = True
    else:
        forced = False

    paired = decide_paired(runs, forced)
    print(f"Using paired tests: {paired} (set --paired true/false to override)")

    # Build tables
    summary_df = build_summary_table(runs, n_boot=int(args.n_boot), alpha=float(args.alpha), rng=rng)
    pairwise_df = build_pairwise_table(
        runs,
        paired=paired,
        n_boot=int(args.n_boot),
        n_perm=int(args.n_perm),
        alpha=float(args.alpha),
        rng=rng,
    )

    summary_df.to_csv(out_dir / "summary_stats.csv", index=False)
    pairwise_df.to_csv(out_dir / "pairwise_comparisons.csv", index=False)

    write_per_joint_table(runs, out_dir)

    # Plots
    plot_metric_distributions(runs, out_dir=out_dir, n_boot=int(args.n_boot), alpha=float(args.alpha), rng=rng, dpi=int(args.dpi))
    plot_pairwise_diffs(pairwise_df, out_dir=out_dir, dpi=int(args.dpi))
    plot_speed_vs_rmse(runs, out_dir=out_dir, dpi=int(args.dpi))

    # Small console summary for quick reading
    key = ["RMSE_deg", "return_env", "speed", "survival"]
    print("\n=== Key means (with 95% bootstrap CI) ===")
    for metric in key:
        if metric not in METRICS:
            continue
        lab = METRICS[metric][0]
        print(f"\n{lab} [{metric}]")
        for rd in runs:
            if metric not in rd.ep.columns:
                continue
            x = rd.ep[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(x, mean, n_boot=int(args.n_boot), alpha=float(args.alpha), rng=rng)
            print(f"  {rd.mode:>8s}: {mean(x): .4f}   (CI {lo: .4f}, {hi: .4f})")

    print(f"\nWrote results to: {out_dir}")


if __name__ == "__main__":
    main()
