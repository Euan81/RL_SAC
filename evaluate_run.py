from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from sac_gaitmatch_clean.train_common import TrainCfg, make_env_baseline, make_env_paper6, make_env_exp6


def mae(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def evaluate(run_dir: Path, *, episodes: int = 20, mode: Optional[str] = None) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Infer mode if not provided
    if mode is None:
        name = run_dir.name.lower()
        if "baseline" in name:
            mode = "baseline"
        elif "paper" in name:
            mode = "paper6"
        elif "exp" in name:
            mode = "exp6"
        else:
            mode = "baseline"

    # Load config if available
    cfg_kwargs: Dict[str, Any] = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg_json = json.loads(cfg_path.read_text())
        for k in ["seed","n_envs","episode_steps","gait_cycle_steps","ref_npz","ref_keys","joint_names","joint_signs","offset_align","left_phase_shift",
                  "paper_scale","paper_margin_deg","paper_w_tr","paper_w_acc","paper_w_lim","paper_acc_total_bonus","paper_lim_total_bonus","paper_env_mix","paper_env_norm","paper_clip","paper_fall_penalty",
                  "exp_alpha","exp_scale","exp_margin_deg","exp_w_tr","exp_w_acc","exp_w_lim","exp_acc_total_bonus","exp_lim_total_bonus","exp_env_mix","exp_env_norm","exp_fall_penalty"]:
            if k in cfg_json:
                cfg_kwargs[k] = cfg_json[k]
        if "paper" in cfg_json:
            p = cfg_json["paper"]
            cfg_kwargs.update({
                "paper_scale": p.get("scale", cfg_kwargs.get("paper_scale")),
                "paper_margin_deg": p.get("margin_deg", cfg_kwargs.get("paper_margin_deg")),
                "paper_w_tr": p.get("w_tr", cfg_kwargs.get("paper_w_tr")),
                "paper_w_acc": p.get("w_acc", cfg_kwargs.get("paper_w_acc")),
                "paper_w_lim": p.get("w_lim", cfg_kwargs.get("paper_w_lim")),
                "paper_acc_total_bonus": p.get("acc_total_bonus", cfg_kwargs.get("paper_acc_total_bonus")),
                "paper_lim_total_bonus": p.get("lim_total_bonus", cfg_kwargs.get("paper_lim_total_bonus")),
                "paper_env_mix": p.get("env_mix", cfg_kwargs.get("paper_env_mix")),
                "paper_env_norm": p.get("env_norm", cfg_kwargs.get("paper_env_norm")),
                "paper_clip": p.get("clip", cfg_kwargs.get("paper_clip")),
                "paper_fall_penalty": p.get("fall_penalty", cfg_kwargs.get("paper_fall_penalty")),
            })
        if "exp" in cfg_json:
            e = cfg_json["exp"]
            cfg_kwargs.update({
                "exp_alpha": e.get("alpha", cfg_kwargs.get("exp_alpha")),
                "exp_scale": e.get("scale", cfg_kwargs.get("exp_scale")),
                "exp_margin_deg": e.get("margin_deg", cfg_kwargs.get("exp_margin_deg")),
                "exp_w_tr": e.get("w_tr", cfg_kwargs.get("exp_w_tr")),
                "exp_w_acc": e.get("w_acc", cfg_kwargs.get("exp_w_acc")),
                "exp_w_lim": e.get("w_lim", cfg_kwargs.get("exp_w_lim")),
                "exp_acc_total_bonus": e.get("acc_total_bonus", cfg_kwargs.get("exp_acc_total_bonus")),
                "exp_lim_total_bonus": e.get("lim_total_bonus", cfg_kwargs.get("exp_lim_total_bonus")),
                "exp_env_mix": e.get("env_mix", cfg_kwargs.get("exp_env_mix")),
                "exp_env_norm": e.get("env_norm", cfg_kwargs.get("exp_env_norm")),
                "exp_fall_penalty": e.get("fall_penalty", cfg_kwargs.get("exp_fall_penalty")),
            })

    cfg = TrainCfg(**cfg_kwargs)

    if mode == "baseline":
        make_env_fn = make_env_baseline
    elif mode in {"paper6","paper"}:
        make_env_fn = make_env_paper6
    elif mode in {"exp6","exp"}:
        make_env_fn = make_env_exp6
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # model path: use base name (SB3 appends .zip)
    if (run_dir / "best" / "best_model.zip").exists():
        model_path = run_dir / "best" / "best_model"
    elif (run_dir / "final_model.zip").exists():
        model_path = run_dir / "final_model"
    else:
        raise FileNotFoundError(f"No model file found in {run_dir} (expected best/best_model.zip or final_model.zip)")

    vec_path = run_dir / "vecnormalize.pkl"

    venv = make_vec_env(lambda: make_env_fn(cfg, seed=cfg.seed + 999), n_envs=1, seed=cfg.seed + 999)
    venv = VecNormalize(venv, training=False, norm_obs=True, norm_reward=False)

    if vec_path.exists():
        venv = VecNormalize.load(str(vec_path), venv)
    venv.training = False
    venv.norm_reward = False

    model = SAC.load(str(model_path), env=venv)

    joint_names = list(cfg.joint_names)

    # Per-episode metrics (for box-plots / robust statistics)
    returns_total, returns_env, lengths = [], [], []
    speeds, ctrl_costs, rmse_deg_ep, mae_deg_ep, survived = [], [], [], [], []
    all_err = []

    # one deterministic rollout for plotting
    q_roll, qref_roll = [], []

    for ep in range(int(episodes)):
        obs = venv.reset()
        done = False
        t = 0

        ep_total = 0.0
        ep_env = 0.0
        ep_err, ep_speed, ep_ctrl = [], [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)
            info = infos[0]

            ep_total += float(info.get("reward_total", rewards[0]))
            ep_env += float(info.get("reward_env", rewards[0]))

            if "q_err" in info:
                ep_err.append(info["q_err"])

            if "x_velocity" in info:
                ep_speed.append(float(info["x_velocity"]))

            if "ctrl_cost" in info:
                ep_ctrl.append(float(info["ctrl_cost"]))
            elif "reward_ctrl" in info:
                rc = float(info["reward_ctrl"])
                ep_ctrl.append(float(-rc) if rc < 0 else float(rc))

            if ep == 0 and ("q" in info and "q_target" in info):
                q_roll.append(info["q"])
                qref_roll.append(info["q_target"])

            done = bool(dones[0])
            t += 1

        lengths.append(t)
        returns_total.append(ep_total)
        returns_env.append(ep_env)

        if len(ep_err):
            all_err.append(np.asarray(ep_err))
        # Episode-aggregated signals
        speeds.append(float(np.mean(ep_speed)) if len(ep_speed) else float("nan"))
        ctrl_costs.append(float(np.mean(ep_ctrl)) if len(ep_ctrl) else float("nan"))
        survived.append(float(t >= int(cfg.episode_steps)))

        if len(ep_err):
            e = np.rad2deg(np.asarray(ep_err, dtype=np.float64))
            rmse_deg_ep.append(rmse(e))
            mae_deg_ep.append(mae(e))
        else:
            rmse_deg_ep.append(float("nan"))
            mae_deg_ep.append(float("nan"))

    if len(all_err):
        all_err = np.concatenate(all_err, axis=0)
        err_deg = np.rad2deg(all_err)
    else:
        err_deg = np.zeros((0, len(joint_names)), dtype=np.float64)

    metrics: Dict[str, Any] = {
        "mode": mode,
        "episodes": int(episodes),
        "return_total_mean": float(np.mean(returns_total)) if len(returns_total) else float("nan"),
        "return_total_std": float(np.std(returns_total)) if len(returns_total) else float("nan"),
        "return_env_mean": float(np.mean(returns_env)) if len(returns_env) else float("nan"),
        "return_env_std": float(np.std(returns_env)) if len(returns_env) else float("nan"),
        "ep_len_mean": float(np.mean(lengths)),
        "ep_len_std": float(np.std(lengths)),
        "survival_rate": float(np.mean(np.asarray(lengths) >= float(cfg.episode_steps))) if len(lengths) else float("nan"),
        "speed_mean": float(np.mean(speeds)) if len(speeds) else float("nan"),
        "speed_std": float(np.std(speeds)) if len(speeds) else float("nan"),
        "ctrl_cost_mean": float(np.mean(ctrl_costs)) if len(ctrl_costs) else float("nan"),
        "ctrl_cost_std": float(np.std(ctrl_costs)) if len(ctrl_costs) else float("nan"),
        "overall": {
            "MAE_deg": mae(err_deg) if err_deg.size else float("nan"),
            "RMSE_deg": rmse(err_deg) if err_deg.size else float("nan"),
        },
        "per_joint": {},
    }

    for j, name in enumerate(joint_names):
        if err_deg.size:
            metrics["per_joint"][name] = {"MAE_deg": mae(err_deg[:, j]), "RMSE_deg": rmse(err_deg[:, j])}
        else:
            metrics["per_joint"][name] = {"MAE_deg": float("nan"), "RMSE_deg": float("nan")}

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save per-episode values for publication-quality plots (box plots etc.)
    try:
        import pandas as pd

        ep_df = pd.DataFrame(
            {
                "return_env": np.asarray(returns_env, dtype=float),
                "return_total": np.asarray(returns_total, dtype=float),
                "ep_len": np.asarray(lengths, dtype=float),
                "survival": np.asarray(survived, dtype=float),
                "speed": np.asarray(speeds, dtype=float),
                "ctrl_cost": np.asarray(ctrl_costs, dtype=float),
                "RMSE_deg": np.asarray(rmse_deg_ep, dtype=float),
                "MAE_deg": np.asarray(mae_deg_ep, dtype=float),
            }
        )
        ep_df.to_csv(out_dir / "episode_stats.csv", index=False)
    except Exception:
        # Plotting scripts can fall back to metrics.json if pandas is unavailable.
        pass

    if len(q_roll):
        np.savez(out_dir / "rollout_joint_angles.npz", q=np.asarray(q_roll, dtype=np.float64), q_ref=np.asarray(qref_roll, dtype=np.float64))

    venv.close()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--mode", choices=["baseline","paper6","exp6"], default=None)
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    m = evaluate(Path(args.run_dir), episodes=args.episodes, mode=args.mode)
    print(json.dumps(m["overall"], indent=2))
    print("return_env_mean:", m["return_env_mean"])
    print("return_total_mean:", m["return_total_mean"])
    print("speed_mean:", m["speed_mean"])
    print("ep_len_mean:", m["ep_len_mean"])
    print("survival_rate:", m["survival_rate"])


if __name__ == "__main__":
    main()
