"""Toy problem: key and door (Step 1 of the proposal).

A 2D point (single integrator, box-bounded velocity) lives in the square room
[-L, L]^2.  A vertical wall at ``x = wall_x`` splits the room; it has a door
gap ``|y| < door_half_width`` that is closed until the point touches the
switch.  The goal is on the far side of the wall, and the cost only measures
the distance to the goal, so a short-horizon planner has no reason to detour
to the switch.

State ``x = (p_x, p_y, k)`` with ``k = 1`` once the switch has been touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from rloc.core.reward_api import ControlEffort, PositionTarget, Reward
from rloc.envs.base import BatchedEnv

Tensor = torch.Tensor


@dataclass
class KeyDoorConfig:
    half_size: float = 1.0
    wall_x: float = 0.0
    door_half_width: float = 0.15
    switch_pos: tuple[float, float] = (-0.6, 0.75)
    switch_radius: float = 0.1
    u_max: float = 1.0
    dt: float = 0.1
    goal_tol: float = 0.05
    # regions used for evaluation episodes
    start_x: tuple[float, float] = (-0.9, -0.1)
    start_y: tuple[float, float] = (-0.9, 0.9)
    goal_x: tuple[float, float] = (0.4, 0.9)
    goal_y: tuple[float, float] = (-0.8, 0.8)


class KeyDoor2D(BatchedEnv):
    nx = 3
    nu = 2
    entities = ("point", "switch")

    def __init__(self, cfg: Optional[KeyDoorConfig] = None):
        self.cfg = cfg or KeyDoorConfig()
        c = self.cfg
        self.dt = c.dt
        self.u_low = torch.full((2,), -c.u_max)
        self.u_high = torch.full((2,), c.u_max)
        self.switch = torch.tensor(c.switch_pos, dtype=torch.float32)
        self._eps = 1e-4  # how far a blocked point stays from the wall

    # ------------------------------------------------------------------ #
    # dynamics
    # ------------------------------------------------------------------ #
    def step(self, x: Tensor, u: Tensor) -> Tensor:
        c = self.cfg
        p, k = x[..., :2], x[..., 2]
        u = u.clamp(-c.u_max, c.u_max)
        q = (p + c.dt * u).clamp(-c.half_size, c.half_size)

        # wall crossing test on the segment p -> q
        left_old = p[..., 0] < c.wall_x
        left_new = q[..., 0] < c.wall_x
        crossed = left_old != left_new
        dx = q[..., 0] - p[..., 0]
        safe_dx = torch.where(dx.abs() < 1e-12, torch.ones_like(dx), dx)
        frac = ((c.wall_x - p[..., 0]) / safe_dx).clamp(0.0, 1.0)
        y_cross = p[..., 1] + frac * (q[..., 1] - p[..., 1])
        through_door = (k > 0.5) & (y_cross.abs() < c.door_half_width)
        blocked = crossed & ~through_door

        # blocked points stop just before the wall but keep their y motion
        x_stop = torch.where(left_old, c.wall_x - self._eps, c.wall_x + self._eps)
        qx = torch.where(blocked, x_stop, q[..., 0])
        q = torch.stack([qx, q[..., 1]], dim=-1)

        touched = (q - self.switch).norm(dim=-1) < c.switch_radius
        k_new = torch.maximum(k, touched.to(k.dtype))
        return torch.cat([q, k_new[..., None]], dim=-1)

    def entity_pos(self, x: Tensor, name: str) -> Tensor:
        if name == "point":
            return x[..., :2]
        if name == "switch":
            return self.switch.expand(*x.shape[:-1], 2)
        raise KeyError(f"unknown entity {name!r}; known: {self.entities}")

    # ------------------------------------------------------------------ #
    # reward
    # ------------------------------------------------------------------ #
    @staticmethod
    def default_reward(norm: str = "quadratic") -> Reward:
        """The 'instruction' of the toy problem: go to the goal, gently."""
        return Reward({"goal": PositionTarget("point", 2, norm), "effort": ControlEffort()})

    @staticmethod
    def make_psi(reward: Reward, goal, goal_weight: float = 1.0, effort_weight: float = 0.01) -> Tensor:
        goal = torch.as_tensor(goal, dtype=torch.float32)
        if goal.ndim == 1:
            return reward.pack({"goal": {"weight": goal_weight, "target": goal},
                                "effort": {"weight": effort_weight}})
        return torch.stack([KeyDoor2D.make_psi(reward, g, goal_weight, effort_weight) for g in goal])

    @staticmethod
    def goal_from_psi(reward: Reward, psi: Tensor) -> Tensor:
        return reward.unpack(psi)["goal"][1]["target"]

    # ------------------------------------------------------------------ #
    # sampling and success
    # ------------------------------------------------------------------ #
    def _uniform(self, n, lo, hi, generator):
        return lo + (hi - lo) * torch.rand(n, generator=generator)

    def sample_states(self, n: int, generator: Optional[torch.Generator] = None,
                      k: Optional[float] = None) -> Tensor:
        """Uniform positions in the room; k ~ Bernoulli(0.5) unless given."""
        L = self.cfg.half_size
        p = self._uniform((n, 2), -L, L, generator)
        if k is None:
            kk = (torch.rand(n, generator=generator) < 0.5).float()
        else:
            kk = torch.full((n,), float(k))
        return torch.cat([p, kk[:, None]], dim=-1)

    def sample_start_states(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        """Evaluation starts: left of the wall, switch not pressed, not on the switch."""
        c = self.cfg
        out = []
        while sum(len(o) for o in out) < n:
            px = self._uniform(n, *c.start_x, generator)
            py = self._uniform(n, *c.start_y, generator)
            p = torch.stack([px, py], -1)
            ok = (p - self.switch).norm(dim=-1) > 2 * c.switch_radius
            out.append(p[ok])
        p = torch.cat(out)[:n]
        return torch.cat([p, torch.zeros(n, 1)], dim=-1)

    def sample_goals(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        c = self.cfg
        gx = self._uniform(n, *c.goal_x, generator)
        gy = self._uniform(n, *c.goal_y, generator)
        return torch.stack([gx, gy], -1)

    def success(self, x: Tensor, goal: Tensor) -> Tensor:
        return (x[..., :2] - goal).norm(dim=-1) < self.cfg.goal_tol
