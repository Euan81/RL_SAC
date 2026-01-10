from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
import imageio.v2 as imageio

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from sac_gaitmatch_clean.train_common import TrainCfg, make_env_baseline, make_env_paper6, make_env_exp6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--format", choices=["mp4","gif"], default="mp4")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.json"
    cfg = TrainCfg()
    mode = None
    if cfg_path.exists():
        import json
        cfg_json = json.loads(cfg_path.read_text())
        mode = cfg_json.get("mode", None)
        # minimal set
        for k in ["seed","episode_steps","gait_cycle_steps","ref_npz","ref_keys","joint_names","joint_signs","offset_align","left_phase_shift",
                  "paper_scale","paper_margin_deg","paper_w_tr","paper_w_acc","paper_w_lim","paper_acc_total_bonus","paper_lim_total_bonus","paper_clip","paper_fall_penalty",
                  "exp_alpha","exp_scale","exp_margin_deg","exp_w_tr","exp_w_acc","exp_w_lim","exp_acc_total_bonus","exp_lim_total_bonus","exp_fall_penalty"]:
            if k in cfg_json:
                setattr(cfg, k, cfg_json[k])

    if mode is None:
        name = run_dir.name.lower()
        if "baseline" in name:
            mode = "baseline"
        elif "paper" in name:
            mode = "paper6"
        else:
            mode = "exp6"

    if mode == "baseline":
        make_env_fn = make_env_baseline
    elif mode == "paper6":
        make_env_fn = make_env_paper6
    else:
        make_env_fn = make_env_exp6

    if (run_dir / "best" / "best_model.zip").exists():
        model_path = run_dir / "best" / "best_model"
    else:
        model_path = run_dir / "final_model"
    vec_path = run_dir / "vecnormalize.pkl"

    # For video we need a non-vectorized env with rgb_array rendering.
    # We'll create a fresh env using gym.make directly for rendering, then wrap similarly.
    # Easiest: use make_env_fn and then access underlying unwrapped env; but SB3 expects VecEnv.
    venv = make_vec_env(
        lambda: make_env_fn(cfg, seed=cfg.seed + 1234, render_mode="rgb_array"),
        n_envs=1,
        seed=cfg.seed + 1234
    )

    venv = VecNormalize(venv, training=False, norm_obs=True, norm_reward=False)
    if vec_path.exists():
        venv = VecNormalize.load(str(vec_path), venv)
    venv.training = False
    venv.norm_reward = False

    model = SAC.load(str(model_path), env=venv)

    # Build a raw env for rendering (not VecEnv)
    env = gym.make("Walker2d-v5", render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=int(cfg.episode_steps))
    # reuse wrappers by calling make_env_fn and stealing them is messy; instead, just record env visuals using actions from model via venv.
    # We'll step the VecEnv and render from the underlying env inside it.
    # SB3's DummyVecEnv stores envs in venv.venv.envs[0]
    try:
        render_env = venv.venv.envs[0]
    except Exception:
        render_env = None

    out_dir = run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval.{args.format}"

    frames = []
    for ep in range(int(args.episodes)):
        obs = venv.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = venv.step(action)
            done = bool(dones[0])
            if render_env is not None and hasattr(render_env, "render"):
                frame = render_env.render()
                if frame is not None:
                    frames.append(frame)

    # Write video
    if args.format == "mp4":
        try:
            with imageio.get_writer(str(out_path), fps=int(args.fps)) as w:
                for f in frames:
                    w.append_data(f)
        except Exception:
            # fallback to gif
            out_path = out_dir / "eval.gif"
            imageio.mimsave(str(out_path), frames, fps=int(args.fps))
    else:
        imageio.mimsave(str(out_path), frames, fps=int(args.fps))

    venv.close()
    print(f"Wrote video: {out_path}")


if __name__ == "__main__":
    main()
