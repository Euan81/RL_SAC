from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from tqdm import tqdm

from sac_gaitmatch_clean.train_common import TrainCfg, make_env_baseline, make_env_paper6, make_env_exp6, save_config


class TqdmProgressCallback:
    """A minimal callback-like object to show progress with tqdm."""

    def __init__(self, total_timesteps: int):
        self.pbar = tqdm(total=total_timesteps, desc="Training", unit="ts", dynamic_ncols=True)
        self.n = 0

    def on_step(self) -> bool:
        # SB3 calls callbacks every step; this keeps it lightweight
        self.n += 1
        if self.n % 100 == 0:
            self.pbar.update(100)
        return True

    def on_training_end(self) -> None:
        remaining = self.pbar.total - self.pbar.n
        if remaining > 0:
            self.pbar.update(remaining)
        self.pbar.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "paper6", "exp6"], required=True)
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--gait-cycle-steps", type=int, default=100)
    ap.add_argument("--ref-npz", type=str, default="references/gait_ref_6d_mujoco.npz")
    ap.add_argument("--device", type=str, default="auto", help="torch device: auto|cpu|cuda|mps")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = TrainCfg(
        seed=int(args.seed),
        n_envs=int(args.n_envs),
        episode_steps=int(args.episode_steps),
        gait_cycle_steps=int(args.gait_cycle_steps),
        ref_npz=str(args.ref_npz),
    )
    save_config(run_dir, cfg, mode=args.mode)

    if args.mode == "baseline":
        make_single_env = lambda seed: make_env_baseline(cfg, seed=seed)
    elif args.mode == "paper6":
        make_single_env = lambda seed: make_env_paper6(cfg, seed=seed)
    else:
        make_single_env = lambda seed: make_env_exp6(cfg, seed=seed)

    # SB3's make_vec_env expects an env id (string) or a callable that takes
    # no required positional arguments. Our env builders require an explicit
    # seed, so we construct the VecEnv manually.
    def _thunk(rank: int):
        return lambda: make_single_env(cfg.seed + rank)

    venv = DummyVecEnv([_thunk(i) for i in range(cfg.n_envs)])
    venv = VecNormalize(venv, training=True, norm_obs=True, norm_reward=False)

    # Eval env must share VecNormalize statistics with the training env, otherwise
    # the policy is evaluated on differently normalised observations.
    eval_env = DummyVecEnv([lambda: make_single_env(cfg.seed + 999)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    try:
        eval_env.obs_rms = venv.obs_rms
    except Exception:
        pass

    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(run_dir / "tb"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = SAC(
        "MlpPolicy",
        venv,
        verbose=1,
        seed=cfg.seed,
        tensorboard_log=str(run_dir / "tb"),
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=10_000,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        device=args.device,
    )

    pbar = TqdmProgressCallback(total_timesteps=int(args.timesteps))

    # We integrate progress updates via SB3 callback API using a thin wrapper.
    from stable_baselines3.common.callbacks import BaseCallback

    class _PbarCB(BaseCallback):
        def _on_step(self) -> bool:
            return pbar.on_step()
        def _on_training_end(self) -> None:
            pbar.on_training_end()

    model.learn(total_timesteps=int(args.timesteps), callback=[eval_cb, _PbarCB()])

    # Save final model and VecNormalize stats (use base name; SB3 appends .zip)
    model.save(str(run_dir / "final_model"))
    venv.save(str(run_dir / "vecnormalize.pkl"))

    venv.close()
    eval_env.close()


if __name__ == "__main__":
    main()
