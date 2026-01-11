from pathlib import Path
import argparse
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def export_all_scalars(logdir: Path) -> pd.DataFrame:
    event_files = list(logdir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found under: {logdir}")

    rows = []
    for ev in event_files:
        ea = event_accumulator.EventAccumulator(str(ev), size_guidance={"scalars": 0})
        ea.Reload()

        run_name = ev.parent.name  # usually run folder name
        for tag in ea.Tags().get("scalars", []):
            for s in ea.Scalars(tag):
                rows.append({
                    "run": run_name,
                    "tag": tag,
                    "step": s.step,
                    "value": s.value,
                    "wall_time": s.wall_time,
                    "event_file": str(ev),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Found event files but no scalar data under: {logdir}")
    return df.sort_values(["tag", "run", "step"]).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, required=True, help="Folder containing TB event files")
    ap.add_argument("--out", type=str, default="tb_scalars_long.csv", help="Output CSV")
    args = ap.parse_args()

    df = export_all_scalars(Path(args.logdir))
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(df)} scalar rows.")
    print("Example tags:", df["tag"].drop_duplicates().head(30).tolist())

if __name__ == "__main__":
    main()
