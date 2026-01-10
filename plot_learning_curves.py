from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ema(x: pd.Series, alpha: float) -> pd.Series:
    return x.ewm(alpha=alpha, adjust=False).mean()

def bootstrap_ci(values: np.ndarray, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return values[0], values[0]
    boots = []
    n = len(values)
    for _ in range(n_boot):
        samp = rng.choice(values, size=n, replace=True)
        boots.append(np.mean(samp))
    lo = np.percentile(boots, 100*(alpha/2))
    hi = np.percentile(boots, 100*(1-alpha/2))
    return lo, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="tb_scalars_long.csv")
    ap.add_argument("--tag", type=str, required=True, help="Scalar tag, e.g. eval/mean_reward")
    ap.add_argument("--out", type=str, default="learning_curve.png")
    ap.add_argument("--alpha", type=float, default=0.05, help="EMA smoothing factor")
    ap.add_argument("--bin", type=int, default=10_000, help="Step bin width for aggregation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["tag"] == args.tag].copy()
    if df.empty:
        raise ValueError(f"No rows found for tag={args.tag}. Available tags include:\n"
                         f"{pd.read_csv(args.csv)['tag'].drop_duplicates().head(50).tolist()}")

    # Bin steps so different runs align even if logged at slightly different timesteps
    df["step_bin"] = (df["step"] // args.bin) * args.bin

    # Optional: infer method from run name (baseline/paper6/exp6)
    def infer_method(run: str) -> str:
        r = run.lower()
        for m in ["baseline", "paper6", "exp6"]:
            if m in r:
                return m
        return "other"

    df["method"] = df["run"].apply(infer_method)

    # Aggregate across runs within each method and step_bin:
    # mean + bootstrap CI across runs
    grouped = []
    for (method, step_bin), g in df.groupby(["method", "step_bin"]):
        vals = g.groupby("run")["value"].mean().values  # one value per run in this bin
        mu = np.mean(vals) if len(vals) else np.nan
        lo, hi = bootstrap_ci(vals, seed=args.seed)
        grouped.append({"method": method, "step": step_bin, "mean": mu, "lo": lo, "hi": hi, "n_runs": len(vals)})
    agg = pd.DataFrame(grouped).sort_values(["method", "step"])

    # Plot
    plt.figure()
    for method, g in agg.groupby("method"):
        if method == "other":
            continue
        g = g.sort_values("step")
        y = pd.Series(g["mean"].values)
        y_s = ema(y, alpha=args.alpha)

        # Smoothed mean line
        plt.plot(g["step"], y_s, label=f"{method} (EMA α={args.alpha})", linewidth=2)

        # CI band (unsmoothed CI; defensible + standard)
        plt.fill_between(g["step"], g["lo"], g["hi"], alpha=0.15)

    plt.xlabel("Environment steps")
    plt.ylabel(args.tag)
    plt.title(f"Learning curve: {args.tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=600)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
