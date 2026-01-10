from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import gymnasium as gym
import numpy as np

from .reference import load_reference_npz, resample_reference
from .envs import (
    RewardInfoWrapper,
    JointRefLogger,
    ObsAugmentFromInfoWrapper,
    PaperTrackingShapingWrapper,
    ExponentialTrackingShapingWrapper,
)


DEFAULT_JOINT_NAMES = ("hip_R","knee_R","ankle_R","hip_L","knee_L","ankle_L")


@dataclass
class TrainCfg:
    seed: int = 0
    n_envs: int = 1
    episode_steps: int = 500
    gait_cycle_steps: int = 100

    # reference
    # Default: reference already expressed in MuJoCo Walker2d joint coordinates
    # (thigh/leg/foot for each leg). This avoids any extra sign hacks.
    ref_npz: str = "references/gait_ref_6d_mujoco.npz"
    ref_keys: Tuple[str, ...] = DEFAULT_JOINT_NAMES

    joint_names: Tuple[str, ...] = DEFAULT_JOINT_NAMES
    # When the reference is in MuJoCo coordinates, the correct convention is
    # typically +1 for all joints.
    joint_signs: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    offset_align: bool = True
    left_phase_shift: bool = True

    # Paper shaping defaults (stable)
    paper_scale: float = 1.0
    paper_margin_deg: float = 2.0
    paper_w_tr: float = 1.0
    # In the paper, all weights are 1.
    paper_w_acc: float = 1.0
    paper_w_lim: float = 1.0
    paper_acc_total_bonus: float = 0.30
    paper_lim_total_bonus: float = 0.30
    # Small mixture of native Walker2d reward (helps prevent "learn to freeze").
    paper_env_mix: float = 0.05
    paper_env_norm: float = 5.0
    paper_clip: bool = True
    paper_fall_penalty: float = 25.0

    # Exponential shaping defaults
    exp_alpha: float = 10.0
    exp_scale: float = 1.0
    exp_margin_deg: float = 2.0
    exp_w_tr: float = 1.0
    exp_w_acc: float = 1.0
    exp_w_lim: float = 1.0
    exp_acc_total_bonus: float = 0.30
    exp_lim_total_bonus: float = 0.30
    exp_env_mix: float = 0.05
    exp_env_norm: float = 5.0
    exp_fall_penalty: float = 25.0


def _make_base_env(cfg, seed, render_mode: str | None = None):
    env = gym.make("Walker2d-v5", render_mode=render_mode)
    env.reset(seed=seed)

    # Override episode length
    env = gym.wrappers.TimeLimit(env, max_episode_steps=int(cfg.episode_steps))

    # Load/prepare reference (one cycle length = gait_cycle_steps)
    ref = load_reference_npz(cfg.ref_npz, cfg.ref_keys)
    ref = resample_reference(ref, int(cfg.gait_cycle_steps))

    # Add logger for q/q_target/q_err (does not change reward or obs)
    env = JointRefLogger(
        env,
        joint_names=cfg.joint_names,
        q_ref=ref,
        signs=cfg.joint_signs,
        offset_align=cfg.offset_align,
        left_phase_shift=cfg.left_phase_shift,
    )

    # Ensure reward_env/reward_total exist for baseline too
    env = RewardInfoWrapper(env)
    return env

def make_env_baseline(cfg, seed, render_mode: str | None = None):
    env = _make_base_env(cfg, seed=seed, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def make_env_paper6(cfg, seed, render_mode: str | None = None):
    env = _make_base_env(cfg, seed=seed, render_mode=render_mode)
    env = PaperTrackingShapingWrapper(
        env,
        joint_names=cfg.joint_names,
        signs=cfg.joint_signs,
        margin_deg=cfg.paper_margin_deg,
        w_tr=cfg.paper_w_tr,
        w_acc=cfg.paper_w_acc,
        w_lim=cfg.paper_w_lim,
        acc_total_bonus=cfg.paper_acc_total_bonus,
        lim_total_bonus=cfg.paper_lim_total_bonus,
        scale=cfg.paper_scale,
        env_mix=cfg.paper_env_mix,
        env_norm=cfg.paper_env_norm,
        fall_penalty=cfg.paper_fall_penalty,
        clip_paper=cfg.paper_clip,
    )
    env = ObsAugmentFromInfoWrapper(env, gait_cycle_steps=cfg.gait_cycle_steps, joint_dim=len(cfg.joint_names))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def make_env_exp6(cfg, seed, render_mode: str | None = None):
    env = _make_base_env(cfg, seed=seed, render_mode=render_mode)
    env = ExponentialTrackingShapingWrapper(
        env,
        joint_names=cfg.joint_names,
        signs=cfg.joint_signs,
        alpha=cfg.exp_alpha,
        margin_deg=cfg.exp_margin_deg,
        w_tr=cfg.exp_w_tr,
        w_acc=cfg.exp_w_acc,
        w_lim=cfg.exp_w_lim,
        acc_total_bonus=cfg.exp_acc_total_bonus,
        lim_total_bonus=cfg.exp_lim_total_bonus,
        scale=cfg.exp_scale,
        env_mix=cfg.exp_env_mix,
        env_norm=cfg.exp_env_norm,
        fall_penalty=cfg.exp_fall_penalty,
    )
    env = ObsAugmentFromInfoWrapper(env, gait_cycle_steps=cfg.gait_cycle_steps, joint_dim=len(cfg.joint_names))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def save_config(run_dir: Path, cfg: TrainCfg, mode: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    d = asdict(cfg)
    d["mode"] = mode
    # structured blocks for human readability
    d["paper"] = {
        "scale": cfg.paper_scale,
        "margin_deg": cfg.paper_margin_deg,
        "w_tr": cfg.paper_w_tr,
        "w_acc": cfg.paper_w_acc,
        "w_lim": cfg.paper_w_lim,
        "acc_total_bonus": cfg.paper_acc_total_bonus,
        "lim_total_bonus": cfg.paper_lim_total_bonus,
        "env_mix": cfg.paper_env_mix,
        "env_norm": cfg.paper_env_norm,
        "clip": cfg.paper_clip,
        "fall_penalty": cfg.paper_fall_penalty,
    }
    d["exp"] = {
        "alpha": cfg.exp_alpha,
        "scale": cfg.exp_scale,
        "margin_deg": cfg.exp_margin_deg,
        "w_tr": cfg.exp_w_tr,
        "w_acc": cfg.exp_w_acc,
        "w_lim": cfg.exp_w_lim,
        "acc_total_bonus": cfg.exp_acc_total_bonus,
        "lim_total_bonus": cfg.exp_lim_total_bonus,
        "env_mix": cfg.exp_env_mix,
        "env_norm": cfg.exp_env_norm,
        "fall_penalty": cfg.exp_fall_penalty,
    }
    (run_dir / "config.json").write_text(json.dumps(d, indent=2))
