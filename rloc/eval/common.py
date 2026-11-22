"""Config loading and closed-loop evaluation shared by the scripts."""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import torch
import yaml

from rloc.core.mppi import MPPI, MPCController, MPPIConfig
from rloc.core.reward_api import Reward
from rloc.envs.keydoor2d import KeyDoor2D, KeyDoorConfig

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_keydoor(cfg: dict) -> tuple[KeyDoor2D, Reward]:
    names = {f.name for f in fields(KeyDoorConfig)}
    env_cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg["env"].items() if k in names}
    env = KeyDoor2D(KeyDoorConfig(**env_cfg))
    reward = KeyDoor2D.default_reward(cfg["reward"].get("norm", "quadratic"))
    return env, reward


def make_psi(cfg: dict, reward: Reward, goals) -> torch.Tensor:
    r = cfg["reward"]
    return KeyDoor2D.make_psi(reward, goals, r["goal_weight"], r["effort_weight"])


def mppi_config(cfg: dict, spec: dict) -> MPPIConfig:
    """Shared sampler settings (cfg["mppi"]) + per-controller horizon/knots/update."""
    m = cfg["mppi"]
    return MPPIConfig(horizon=spec["horizon"], num_samples=m["num_samples"], num_iters=m["num_iters"],
                      noise_sigma=m["noise_sigma"], temperature=m["temperature"],
                      num_knots=spec.get("num_knots"), update=spec.get("update", m.get("update", "mppi")))


def build_controller(env, reward: Reward, cfg: dict, spec: dict, terminal=None) -> MPCController:
    """Receding-horizon controller; ``terminal(x, psi)`` is an optional terminal cost."""
    stage = lambda x, u, p: reward.stage_cost(env, x, u, p)  # noqa: E731
    mppi = MPPI(env.step, stage, env.u_low, env.u_high, mppi_config(cfg, spec), terminal_cost_fn=terminal,
                generator=seeded(cfg["eval"]["seed"] + 1))
    return MPCController(mppi)


@dataclass
class EpisodeStats:
    traj: torch.Tensor  # [B, T+1, nx]
    success: torch.Tensor  # [B] bool
    steps: torch.Tensor  # [B] steps until success (max_steps if failed)
    switch: torch.Tensor  # [B] bool, switch pressed at some point
    cost: torch.Tensor  # [B] accumulated stage cost until success / end

    def summary(self) -> dict:
        s = self.success
        return {
            "success_rate": s.float().mean().item(),
            "switch_rate": self.switch.float().mean().item(),
            "mean_steps_success": self.steps[s].float().mean().item() if s.any() else float("nan"),
            "mean_cost": self.cost.mean().item(),
        }


@torch.no_grad()
def run_episodes(env: KeyDoor2D, reward: Reward, controller: MPCController, x0: torch.Tensor,
                 psi: torch.Tensor, max_steps: int) -> EpisodeStats:
    """Run B episodes in parallel (each with its own psi) until success or max_steps."""
    goal = KeyDoor2D.goal_from_psi(reward, psi)
    B = x0.shape[0]
    controller.reset()
    x = x0.clone()
    done = env.success(x, goal)
    steps = torch.full((B,), max_steps)
    steps[done] = 0
    cost = torch.zeros(B)
    traj = [x]
    for t in range(max_steps):
        u = controller.act(x, psi)
        xn = env.step(x, u)
        c = reward.stage_cost(env, xn, u, psi)
        cost = cost + torch.where(done, torch.zeros_like(c), c)
        x = torch.where(done[:, None], x, xn)
        newly = env.success(x, goal) & ~done
        steps[newly] = t + 1
        done = done | newly
        traj.append(x)
        if done.all():
            break
    traj = torch.stack(traj, 1)
    return EpisodeStats(traj=traj, success=done, steps=steps, switch=traj[:, :, 2].amax(1) > 0.5, cost=cost)


@torch.no_grad()
def time_per_step(controller: MPCController, env: KeyDoor2D, x0: torch.Tensor, psi: torch.Tensor,
                  n_steps: int) -> float:
    """Wall-clock seconds per control step for a single episode (B = 1)."""
    controller.reset()
    x, p = x0[:1].clone(), psi[:1]
    controller.act(x, p)  # warm-up
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x = env.step(x, controller.act(x, p))
    return (time.perf_counter() - t0) / n_steps


def seeded(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def maybe_mkdir(path: Optional[Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
