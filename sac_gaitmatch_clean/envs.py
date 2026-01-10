from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym

from .mujoco_utils import get_joint_angles, get_joint_limits


class RewardInfoWrapper(gym.Wrapper):
    """Ensure reward_env/reward_total are always present in info."""

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        info = dict(info) if isinstance(info, dict) else {}
        info.setdefault("reward_env", float(r))
        info.setdefault("reward_total", float(r))
        return obs, r, terminated, truncated, info


class JointRefLogger(gym.Wrapper):
    """Log joint tracking signals (does not change reward or observation).

    Adds to info:
      t, q, q_target, q_err

    Supports a left-leg 50% phase shift by using per-joint phase offsets when J==6.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        joint_names: Sequence[str],
        q_ref: np.ndarray,
        signs: Optional[Sequence[float]] = None,
        offset_align: bool = True,
        left_phase_shift: bool = True,
    ):
        super().__init__(env)
        self.joint_names = tuple(joint_names)
        self.q_ref = np.asarray(q_ref, dtype=np.float64)
        self.J = len(self.joint_names)

        if signs is None:
            self.signs = np.ones(self.J, dtype=np.float64)
        else:
            s = np.asarray(signs, dtype=np.float64)
            if s.shape != (self.J,):
                raise ValueError(f"signs must have shape ({self.J},), got {s.shape}")
            self.signs = s

        self.offset_align = bool(offset_align)
        self.left_phase_shift = bool(left_phase_shift)
        self.t = 0
        self._offset = np.zeros(self.J, dtype=np.float64)

    def _phase_offsets(self) -> np.ndarray:
        P = int(len(self.q_ref))
        if self.left_phase_shift and self.J == 6:
            half = P // 2
            # [hip_R,knee_R,ankle_R, hip_L,knee_L,ankle_L]
            return np.array([0, 0, 0, half, half, half], dtype=int)
        return np.zeros(self.J, dtype=int)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info) if isinstance(info, dict) else {}
        self.t = 0

        q = get_joint_angles(self.env, self.joint_names) * self.signs
        idxs = (self.t + self._phase_offsets()) % len(self.q_ref)
        qdes0 = np.array([self.q_ref[idxs[j], j] for j in range(self.J)], dtype=np.float64)
        self._offset = (q - qdes0) if self.offset_align else np.zeros_like(self._offset)

        qdes = qdes0 + self._offset
        info.update({"t": int(self.t), "q": q.copy(), "q_target": qdes.copy(), "q_err": (q - qdes).copy()})
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info) if isinstance(info, dict) else {}
        self.t += 1

        q = get_joint_angles(self.env, self.joint_names) * self.signs
        idxs = (self.t + self._phase_offsets()) % len(self.q_ref)
        qdes = np.array([self.q_ref[idxs[j], j] for j in range(self.J)], dtype=np.float64) + self._offset

        info.update({"t": int(self.t), "q": q.copy(), "q_target": qdes.copy(), "q_err": (q - qdes).copy()})
        return obs, reward, terminated, truncated, info


class ObsAugmentFromInfoWrapper(gym.Wrapper):
    """Append phase + q_target + q_err to the original observation.

    This is critical for tracking: it provides the policy with the time-varying target.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        gait_cycle_steps: int,
        joint_dim: int,
        add_phase: bool = True,
        add_q_target: bool = True,
        add_q_err: bool = True,
    ):
        super().__init__(env)
        self.P = int(gait_cycle_steps)
        self.joint_dim = int(joint_dim)
        self.add_phase = bool(add_phase)
        self.add_q_target = bool(add_q_target)
        self.add_q_err = bool(add_q_err)

        if not isinstance(env.observation_space, gym.spaces.Box) or env.observation_space.shape is None:
            raise ValueError("ObsAugmentFromInfoWrapper expects a Box observation space")

        base_dim = int(np.prod(env.observation_space.shape))
        extra = 0
        if self.add_phase:
            extra += 2
        if self.add_q_target:
            extra += self.joint_dim
        if self.add_q_err:
            extra += self.joint_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_dim + extra,), dtype=np.float32
        )

    def _augment(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        extras = []
        t = int(info.get("t", 0))
        if self.add_phase:
            phi = 2.0 * np.pi * (float(t % max(1, self.P)) / float(max(1, self.P)))
            extras.append(np.array([np.sin(phi), np.cos(phi)], dtype=np.float32))
        if self.add_q_target:
            extras.append(np.asarray(info["q_target"], dtype=np.float32).reshape(-1))
        if self.add_q_err:
            extras.append(np.asarray(info["q_err"], dtype=np.float32).reshape(-1))
        if extras:
            return np.concatenate([obs, *extras], axis=0)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info) if isinstance(info, dict) else {}
        return self._augment(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info) if isinstance(info, dict) else {}
        return self._augment(obs, info), reward, terminated, truncated, info


class PaperTrackingShapingWrapper(gym.Wrapper):
    """Paper-style gait tracking reward (with an optional small mix of Walker2d reward).

    The paper’s core idea is to:
      * penalise tracking error (negative term),
      * provide small positive bonuses for being within a tight margin and within joint limits,
      * keep the overall reward bounded.

    For Walker2d we additionally allow a tiny mixture of the native environment reward
    (scaled + bounded) to help maintain stable forward locomotion.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        joint_names: Sequence[str],
        weights: Optional[Sequence[float]] = None,
        signs: Optional[Sequence[float]] = None,
        margin_deg: float = 2.0,
        w_tr: float = 1.0,
        w_acc: float = 0.5,
        w_lim: float = 0.2,
        acc_total_bonus: float = 0.30,
        lim_total_bonus: float = 0.10,
        scale: float = 1.0,
        # How much of the native Walker2d reward to keep.
        # 0.0 -> pure tracking reward (paper-style)
        # 1.0 -> pure environment reward
        env_mix: float = 0.05,
        # Normalisation for the environment reward before mixing (tanh(r/env_norm)).
        env_norm: float = 5.0,
        fall_penalty: float = 25.0,
        clip_paper: bool = True,
    ):
        super().__init__(env)
        self.joint_names = tuple(joint_names)
        self.J = len(self.joint_names)

        if signs is None:
            self.signs = np.ones(self.J, dtype=np.float64)
        else:
            s = np.asarray(signs, dtype=np.float64)
            if s.shape != (self.J,):
                raise ValueError(f"signs must have shape ({self.J},), got {s.shape}")
            self.signs = s

        if weights is None:
            w = np.ones(self.J, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (self.J,):
                raise ValueError(f"weights must have shape ({self.J},), got {w.shape}")
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
        if float(np.sum(w)) > 0:
            w = w / float(np.sum(w))
        self.wj = w

        qmin, qmax = get_joint_limits(self.env, self.joint_names)
        qmin_s = qmin * self.signs
        qmax_s = qmax * self.signs
        self.qmin = np.minimum(qmin_s, qmax_s)
        self.qmax = np.maximum(qmin_s, qmax_s)
        rng = self.qmax - self.qmin
        rng[rng < 1e-6] = 2.0 * np.pi
        self.rng = rng

        self.margin_rad = float(np.deg2rad(margin_deg))
        self.w_tr = float(w_tr)
        self.w_acc = float(w_acc)
        self.w_lim = float(w_lim)
        self.acc_per_joint = float(acc_total_bonus) / float(self.J)
        self.lim_per_joint = float(lim_total_bonus) / float(self.J)

        self.scale = float(scale)
        self.env_mix = float(env_mix)
        self.env_norm = float(env_norm)
        self.fall_penalty = float(fall_penalty)
        self.clip_paper = bool(clip_paper)

    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)
        info = dict(info) if isinstance(info, dict) else {}

        if "q" in info and "q_target" in info:
            q = np.asarray(info["q"], dtype=np.float64)
            qdes = np.asarray(info["q_target"], dtype=np.float64)
        else:
            q = get_joint_angles(self.env, self.joint_names) * self.signs
            qdes = q

        # --- Paper reward (bounded) ---
        # Trajectory term: negative normalised absolute error in [-1, 0]
        abs_err = np.abs(q - qdes)
        norm_err = np.clip(abs_err / self.rng, 0.0, 1.0)
        mean_norm_err = float(np.sum(self.wj * norm_err))
        r_tr = float(-mean_norm_err)  # [-1, 0]

        within = (abs_err <= self.margin_rad)
        r_acc = float(np.sum(within.astype(np.float64) * self.acc_per_joint))

        in_lim = np.logical_and(q >= self.qmin, q <= self.qmax)
        r_lim = float(np.sum(in_lim.astype(np.float64) * self.lim_per_joint))

        r_paper = (self.w_tr * r_tr) + (self.w_acc * r_acc) + (self.w_lim * r_lim)
        if self.clip_paper:
            r_paper = float(np.clip(r_paper, -1.0, 1.0))

        # Mix with native Walker2d reward (scaled + bounded) so that tracking
        # dominates but the policy still prefers stable forward locomotion.
        env_scaled = float(np.tanh(float(r_env) / max(1e-6, self.env_norm)))
        mix = float(np.clip(self.env_mix, 0.0, 1.0))
        r = float(((1.0 - mix) * (self.scale * r_paper)) + (mix * env_scaled))

        fell = bool(terminated and (not truncated))
        if fell:
            r -= self.fall_penalty

        info.update({
            "reward_env": float(r_env),
            "reward_total": float(r),
            "paper_r": float(r_paper),
            "paper_rtr": float(r_tr),
            "paper_mean_norm_err": float(mean_norm_err),
            "paper_racc": float(r_acc),
            "paper_rlim": float(r_lim),
            "env_reward_scaled": float(env_scaled),
            "env_mix": float(mix),
            "fell": float(fell),
        })
        return obs, r, terminated, truncated, info


class ExponentialTrackingShapingWrapper(gym.Wrapper):
    """Improved shaping: exponential tracking score + small paper-inspired bonuses."""

    def __init__(
        self,
        env: gym.Env,
        *,
        joint_names: Sequence[str],
        weights: Optional[Sequence[float]] = None,
        signs: Optional[Sequence[float]] = None,
        # Controls how quickly the exponential score decays with tracking error.
        # Too large -> score ~0 almost always (no learning signal).
        alpha: float = 10.0,
        margin_deg: float = 2.0,
        w_tr: float = 1.0,
        w_acc: float = 0.3,
        w_lim: float = 0.2,
        acc_total_bonus: float = 0.20,
        lim_total_bonus: float = 0.10,
        scale: float = 1.0,
        env_mix: float = 0.05,
        env_norm: float = 5.0,
        fall_penalty: float = 25.0,
    ):
        super().__init__(env)
        self.joint_names = tuple(joint_names)
        self.J = len(self.joint_names)

        if signs is None:
            self.signs = np.ones(self.J, dtype=np.float64)
        else:
            s = np.asarray(signs, dtype=np.float64)
            if s.shape != (self.J,):
                raise ValueError(f"signs must have shape ({self.J},), got {s.shape}")
            self.signs = s

        if weights is None:
            w = np.ones(self.J, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (self.J,):
                raise ValueError(f"weights must have shape ({self.J},), got {w.shape}")
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
        if float(np.sum(w)) > 0:
            w = w / float(np.sum(w))
        self.wj = w

        qmin, qmax = get_joint_limits(self.env, self.joint_names)
        qmin_s = qmin * self.signs
        qmax_s = qmax * self.signs
        self.qmin = np.minimum(qmin_s, qmax_s)
        self.qmax = np.maximum(qmin_s, qmax_s)
        rng = self.qmax - self.qmin
        rng[rng < 1e-6] = 2.0 * np.pi
        self.rng = rng

        self.alpha = float(alpha)
        self.margin_rad = float(np.deg2rad(margin_deg))
        self.w_tr = float(w_tr)
        self.w_acc = float(w_acc)
        self.w_lim = float(w_lim)
        self.acc_per_joint = float(acc_total_bonus) / float(self.J)
        self.lim_per_joint = float(lim_total_bonus) / float(self.J)
        self.scale = float(scale)
        self.env_mix = float(env_mix)
        self.env_norm = float(env_norm)
        self.fall_penalty = float(fall_penalty)

    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)
        info = dict(info) if isinstance(info, dict) else {}

        if "q" in info and "q_target" in info:
            q = np.asarray(info["q"], dtype=np.float64)
            qdes = np.asarray(info["q_target"], dtype=np.float64)
        else:
            q = get_joint_angles(self.env, self.joint_names) * self.signs
            qdes = q

        # --- Exponential tracking ---
        e = (q - qdes)
        norm = e / self.rng
        mse = float(np.sum(self.wj * (norm * norm)))
        # Score in (0, 1]
        track_score = float(np.exp(-self.alpha * mse))
        # Reward term: use log(score) = -alpha*mse for a strong, non-vanishing
        # gradient, then clip to keep the overall reward bounded.
        r_tr = float(np.clip(-self.alpha * mse, -1.0, 0.0))

        abs_err = np.abs(e)
        within = (abs_err <= self.margin_rad)
        r_acc = float(np.sum(within.astype(np.float64) * self.acc_per_joint))

        in_lim = np.logical_and(q >= self.qmin, q <= self.qmax)
        r_lim = float(np.sum(in_lim.astype(np.float64) * self.lim_per_joint))

        r_custom = (self.w_tr * r_tr) + (self.w_acc * r_acc) + (self.w_lim * r_lim)
        r_custom = float(np.clip(r_custom, -1.0, 1.0))

        env_scaled = float(np.tanh(float(r_env) / max(1e-6, self.env_norm)))
        mix = float(np.clip(self.env_mix, 0.0, 1.0))
        r = float(((1.0 - mix) * (self.scale * r_custom)) + (mix * env_scaled))

        fell = bool(terminated and (not truncated))
        if fell:
            r -= self.fall_penalty

        info.update({
            "reward_env": float(r_env),
            "reward_total": float(r),
            "exp_track": float(track_score),
            "exp_mse": float(mse),
            "exp_rtr": float(r_tr),
            "exp_racc": float(r_acc),
            "exp_rlim": float(r_lim),
            "env_reward_scaled": float(env_scaled),
            "env_mix": float(mix),
            "fell": float(fell),
        })
        return obs, r, terminated, truncated, info
