from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluate_run import evaluate


def _load_or_eval(run_dir: Path, episodes: int) -> Dict[str, Any]:
    mpath = run_dir / "evaluation" / "metrics.json"
    if mpath.exists():
        return json.loads(mpath.read_text())
    return evaluate(run_dir, episodes=episodes, mode=None)


def _load_episode_stats(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "evaluation" / "episode_stats.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def _savefig(name: str, out_dir: Path, dpi: int) -> None:
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.png", dpi=int(dpi))
    plt.savefig(out_dir / f"{name}.pdf")
    plt.close()


def _boxplot_with_std(
    *,
    data: List[np.ndarray],
    labels: List[str],
    title: str,
    ylabel: str,
    out_dir: Path,
    fname: str,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(6.8, 4.4))
    bp = plt.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=True,
        showfliers=False,
    )
    # Overlay mean ± std (single black errorbar per box)
    means = np.array([float(np.nanmean(x)) for x in data], dtype=float)
    stds = np.array([float(np.nanstd(x, ddof=1)) if np.sum(np.isfinite(x)) > 1 else 0.0 for x in data], dtype=float)
    xs = np.arange(1, len(labels) + 1)
    plt.errorbar(xs, means, yerr=stds, fmt="none", capsize=5)
    plt.title(title)
    plt.ylabel(ylabel)
    _savefig(fname, out_dir, dpi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out-dir", type=str, default="runs/conference_compare")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    ep_rows: List[Tuple[str, pd.DataFrame]] = []
    rollouts: List[Tuple[str, Path]] = []

    for r in args.runs:
        run_dir = Path(r)
        m = _load_or_eval(run_dir, episodes=int(args.episodes))
        mode = m.get("mode", run_dir.name)

        rows.append(
            {
                "run": run_dir.name,
                "mode": mode,
                "return_env_mean": m["return_env_mean"],
                "return_env_std": m["return_env_std"],
                "return_total_mean": m["return_total_mean"],
                "return_total_std": m["return_total_std"],
                "ep_len_mean": m["ep_len_mean"],
                "ep_len_std": m["ep_len_std"],
                "survival_rate": m["survival_rate"],
                "speed_mean": m["speed_mean"],
                "speed_std": m["speed_std"],
                "ctrl_cost_mean": m["ctrl_cost_mean"],
                "ctrl_cost_std": m["ctrl_cost_std"],
                "MAE_deg": m["overall"]["MAE_deg"],
                "RMSE_deg": m["overall"]["RMSE_deg"],
            }
        )

        ep = _load_episode_stats(run_dir)
        if ep is not None:
            ep_rows.append((mode, ep))

        rnpz = run_dir / "evaluation" / "rollout_joint_angles.npz"
        if rnpz.exists():
            rollouts.append((mode, rnpz))

    df = pd.DataFrame(rows)
    order = {"baseline": 0, "paper6": 1, "exp6": 2}
    df["__o"] = df["mode"].map(lambda x: order.get(str(x), 99))
    df = df.sort_values("__o").drop(columns="__o")

    (out_dir / "conference_metrics_table.csv").write_text(df.to_csv(index=False))

    # LaTeX table (booktabs). Raw strings to avoid escape warnings.
    def fmt(ms: float, ss: float, d: int = 2) -> str:
        return f"{ms:.{d}f} $\\pm$ {ss:.{d}f}"

    tex: List[str] = []
    tex += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Walker2d-v5: baseline vs shaping variants (mean $\pm$ std over evaluation episodes).}",
        r"\label{tab:walker2d}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Env return & Total return & Speed (m/s) & Ep len & Survival & RMSE (deg) \\ ",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        tex += [
            f"{row['mode']} & {fmt(row['return_env_mean'], row['return_env_std'],2)} & {fmt(row['return_total_mean'], row['return_total_std'],2)} & {fmt(row['speed_mean'], row['speed_std'],3)} & {fmt(row['ep_len_mean'], row['ep_len_std'],1)} & {row['survival_rate']*100:.1f}\\% & {row['RMSE_deg']:.2f} \\"
        ]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / "conference_metrics_table.tex").write_text("\n".join(tex))

    # If we have per-episode stats, generate publication-friendly *box* plots.
    if ep_rows:
        # Ensure same order as df
        mode_order = df["mode"].tolist()
        ep_map = {m: e for m, e in ep_rows}

        def get_series(col: str) -> List[np.ndarray]:
            out = []
            for m in mode_order:
                e = ep_map.get(m)
                if e is None or col not in e.columns:
                    out.append(np.array([np.nan]))
                else:
                    out.append(np.asarray(e[col].to_numpy(dtype=float)))
            return out

        _boxplot_with_std(
            data=get_series("return_env"),
            labels=mode_order,
            title="Env return (per-episode)",
            ylabel="episodic env return",
            out_dir=out_dir,
            fname="conference_return_env_box",
            dpi=int(args.dpi),
        )
        _boxplot_with_std(
            data=get_series("return_total"),
            labels=mode_order,
            title="Total return (env + shaping, per-episode)",
            ylabel="episodic total return",
            out_dir=out_dir,
            fname="conference_return_total_box",
            dpi=int(args.dpi),
        )
        _boxplot_with_std(
            data=get_series("RMSE_deg"),
            labels=mode_order,
            title="Tracking RMSE (per-episode)",
            ylabel="RMSE (deg)",
            out_dir=out_dir,
            fname="conference_rmse_box",
            dpi=int(args.dpi),
        )
        _boxplot_with_std(
            data=get_series("ep_len"),
            labels=mode_order,
            title="Episode length (per-episode)",
            ylabel="steps",
            out_dir=out_dir,
            fname="conference_ep_len_box",
            dpi=int(args.dpi),
        )
        _boxplot_with_std(
            data=get_series("speed"),
            labels=mode_order,
            title="Forward speed (per-episode)",
            ylabel="m/s",
            out_dir=out_dir,
            fname="conference_speed_box",
            dpi=int(args.dpi),
        )
        _boxplot_with_std(
            data=[x * 100.0 for x in get_series("survival")],
            labels=mode_order,
            title="Survival (per-episode)",
            ylabel="% episodes reaching time-limit",
            out_dir=out_dir,
            fname="conference_survival_box",
            dpi=int(args.dpi),
        )

        # Speed–tracking trade-off scatter with mean±std bars (more informative than bar chart).
        speed = get_series("speed")
        rmse = get_series("RMSE_deg")
        sp_mu = np.array([float(np.nanmean(x)) for x in speed])
        sp_sd = np.array([float(np.nanstd(x, ddof=1)) if np.sum(np.isfinite(x)) > 1 else 0.0 for x in speed])
        rm_mu = np.array([float(np.nanmean(x)) for x in rmse])
        rm_sd = np.array([float(np.nanstd(x, ddof=1)) if np.sum(np.isfinite(x)) > 1 else 0.0 for x in rmse])

        plt.figure(figsize=(6.8, 4.4))
        plt.errorbar(sp_mu, rm_mu, xerr=sp_sd, yerr=rm_sd, fmt="o", capsize=5)
        for i, m in enumerate(mode_order):
            plt.text(sp_mu[i], rm_mu[i], f" {m}", va="center")
        plt.xlabel("Mean forward speed (m/s)")
        plt.ylabel("Tracking RMSE (deg)")
        plt.title("Speed–tracking trade-off (mean ± std)")
        _savefig("conference_speed_rmse", out_dir, int(args.dpi))
    else:
        # Fallback: old-style bars from metrics.json
        modes = df["mode"].tolist()
        x = np.arange(len(modes))

        plt.figure(figsize=(6.8, 4.4))
        plt.bar(x, df["return_env_mean"].values, yerr=df["return_env_std"].values, capsize=4)
        plt.xticks(x, modes)
        plt.ylabel("Env episodic return")
        plt.title("Env return (mean ± std)")
        _savefig("conference_return_env", out_dir, int(args.dpi))

        plt.figure(figsize=(6.8, 4.4))
        plt.bar(x, df["return_total_mean"].values, yerr=df["return_total_std"].values, capsize=4)
        plt.xticks(x, modes)
        plt.ylabel("Total episodic return")
        plt.title("Total return (mean ± std)")
        _savefig("conference_return_total", out_dir, int(args.dpi))

    # Joint overlay plot (one shared reference per joint; hips row, then knees, then ankles).
    if rollouts:
        # Order by mode for determinism
        rollouts_sorted = sorted(rollouts, key=lambda x: {"baseline": 0, "paper6": 1, "exp6": 2}.get(x[0], 99))

        # Use the first rollout's reference as the *single* reference trace.
        d0 = np.load(rollouts_sorted[0][1])
        qref0 = np.asarray(d0["q_ref"], dtype=np.float64)

        joint_layout = [
            ("hip_R", 0), ("hip_L", 3),
            ("knee_R", 1), ("knee_L", 4),
            ("ankle_R", 2), ("ankle_L", 5),
        ]

        # Collect all traces to set a unified y-scale across the figure.
        y_all: List[np.ndarray] = []
        y_all.append(np.rad2deg(qref0))
        for _, p in rollouts_sorted:
            dd = np.load(p)
            y_all.append(np.rad2deg(np.asarray(dd["q"], dtype=np.float64)))
        ycat = np.concatenate([y.reshape(-1, y.shape[-1]) for y in y_all], axis=0)
        ylo, yhi = float(np.nanmin(ycat)), float(np.nanmax(ycat))
        pad = 0.05 * (yhi - ylo + 1e-9)
        ylo, yhi = ylo - pad, yhi + pad

        fig, axes = plt.subplots(3, 2, figsize=(10.4, 8.2), sharex=True, sharey=True)
        axes = axes.flatten()

        for ax, (jn, jidx) in zip(axes, joint_layout):
            Tref = qref0.shape[0]
            ax.plot(np.rad2deg(qref0[:Tref, jidx]), linestyle="--", label="reference")
            for mode, p in rollouts_sorted:
                dd = np.load(p)
                q = np.asarray(dd["q"], dtype=np.float64)
                T = min(q.shape[0], Tref)
                ax.plot(np.rad2deg(q[:T, jidx]), label=mode)
            ax.set_title(jn)
            ax.set_ylim([ylo, yhi])
            ax.set_ylabel("deg")

        axes[-1].set_xlabel("t (steps)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(out_dir / "joint_tracking_compare.png", dpi=int(args.dpi), bbox_inches="tight")
        fig.savefig(out_dir / "joint_tracking_compare.pdf", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
