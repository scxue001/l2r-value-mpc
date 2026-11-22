"""Exact (grid) value function for the key-door toy problem.

The state is (p, k) with p in a 2D box and k in {0, 1}, so the infinite-horizon
value can be computed by semi-Lagrangian value iteration on a grid:

    V(s) = min_a [ stage_cost(s', a) + V(s') ],   s' = step(s, a),

with V(s') interpolated bilinearly.  This is the ground truth that the learned
value function will be compared against, and it gives an "oracle" terminal
cost for MPPI.

Two details make the grid solution faithful to the continuous problem:

* Goal region absorbing.  On a grid the point cannot sit exactly on the goal,
  so the undiscounted sum would grow forever by a tiny amount.  States within
  ``goal_tol`` of the goal are therefore terminal with V = 0 (the stage cost
  there is O(goal_tol^2) per step, negligible).
* No interpolation across the wall.  Query points are clamped to the last grid
  column on their own side of the wall, so values never leak through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from rloc.core.reward_api import Reward
from rloc.envs.keydoor2d import KeyDoor2D

Tensor = torch.Tensor


@dataclass
class GridSpec:
    h: float = 0.02  # grid spacing
    actions_per_axis: int = 9  # control grid in [-u_max, u_max]^2


class GridValue:
    """Bilinear interpolation of a grid value V[k, i, j] (i: x index, j: y index)."""

    def __init__(self, env: KeyDoor2D, V: Tensor, h: float):
        self.env = env
        self.V = V  # [2, N, N]
        self.h = h
        L = env.cfg.half_size
        self.N = V.shape[-1]
        self.lo = -L + h / 2  # first node coordinate
        self.coords = self.lo + h * torch.arange(self.N, dtype=torch.float32)
        w = env.cfg.wall_x
        # last node strictly left of the wall / first node strictly right of it
        self.x_left_max = self.coords[self.coords < w].max().item()
        self.x_right_min = self.coords[self.coords > w].min().item()

    def _corners(self, x: Tensor):
        """Corner flat indices [..., 4] and weights [..., 4] for states x [..., 3]."""
        w = self.env.cfg.wall_x
        px, py, k = x[..., 0], x[..., 1], (x[..., 2] > 0.5).long()
        px = torch.where(px < w, px.clamp(max=self.x_left_max), px.clamp(min=self.x_right_min))
        hi = self.lo + self.h * (self.N - 1)
        px, py = px.clamp(self.lo, hi), py.clamp(self.lo, hi)
        fx, fy = (px - self.lo) / self.h, (py - self.lo) / self.h
        i0 = fx.floor().long().clamp(0, self.N - 2)
        j0 = fy.floor().long().clamp(0, self.N - 2)
        ax, ay = fx - i0, fy - j0
        base = k * self.N * self.N
        idx = torch.stack([
            base + i0 * self.N + j0,
            base + (i0 + 1) * self.N + j0,
            base + i0 * self.N + (j0 + 1),
            base + (i0 + 1) * self.N + (j0 + 1),
        ], dim=-1)
        wts = torch.stack([(1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay], dim=-1)
        return idx, wts

    def interp_flat(self, Vflat: Tensor, idx: Tensor, wts: Tensor) -> Tensor:
        return (Vflat[idx] * wts).sum(-1)

    def __call__(self, x: Tensor, psi: Optional[Tensor] = None) -> Tensor:
        """Value at states x [..., 3]; psi is ignored (V is for one fixed goal)."""
        idx, wts = self._corners(x)
        return self.interp_flat(self.V.reshape(-1), idx, wts)

    def grid_states(self) -> Tensor:
        """All grid states [2*N*N, 3] in the flat order of V."""
        X, Y = torch.meshgrid(self.coords, self.coords, indexing="ij")
        p = torch.stack([X.reshape(-1), Y.reshape(-1)], -1)
        return torch.cat([
            torch.cat([p, torch.zeros(len(p), 1)], -1),
            torch.cat([p, torch.ones(len(p), 1)], -1),
        ])


class KeyDoorDP:
    """Precomputes transitions once, then runs Bellman sweeps as gathers."""

    def __init__(self, env: KeyDoor2D, reward: Reward, psi: Tensor, grid: GridSpec = GridSpec()):
        self.env, self.reward, self.psi, self.grid = env, reward, psi, grid
        L = env.cfg.half_size
        N = int(round(2 * L / grid.h))
        self.N = N
        self.value = GridValue(env, torch.zeros(2, N, N), grid.h)

        a = torch.linspace(-env.cfg.u_max, env.cfg.u_max, grid.actions_per_axis)
        A1, A2 = torch.meshgrid(a, a, indexing="ij")
        self.actions = torch.stack([A1.reshape(-1), A2.reshape(-1)], -1)  # [A, 2]

        S = self.value.grid_states()  # [M, 3]
        M, A = len(S), len(self.actions)
        Sx = S[:, None, :].expand(M, A, 3)
        U = self.actions[None].expand(M, A, 2)
        S_next = env.step(Sx, U)
        self.cost = reward.stage_cost(env, S_next, U, psi.expand(M, A, -1))  # [M, A]
        self.idx, self.wts = self.value._corners(S_next)  # [M, A, 4]
        goal = KeyDoor2D.goal_from_psi(reward, psi)
        self.terminal = env.success(S, goal)  # [M]
        self.S = S

    def bellman(self, Vflat: Tensor) -> tuple[Tensor, Tensor]:
        """One sweep. Returns new V [M] and argmin action index [M]."""
        q = self.cost + self.value.interp_flat(Vflat, self.idx, self.wts)
        Vn, a = q.min(dim=1)
        Vn = torch.where(self.terminal, torch.zeros_like(Vn), Vn)
        return Vn, a

    def solve(self, tol: float = 1e-6, max_iters: int = 5000, verbose: bool = False) -> GridValue:
        """Infinite-horizon value iteration until max |ΔV| < tol."""
        V = torch.zeros(len(self.S))
        for it in range(max_iters):
            Vn, _ = self.bellman(V)
            delta = (Vn - V).abs().max().item()
            V = Vn
            if verbose and it % 100 == 0:
                print(f"  sweep {it:5d}  max|dV| = {delta:.2e}")
            if delta < tol:
                break
        self.iterations = it + 1
        return GridValue(self.env, V.reshape(2, self.N, self.N), self.grid.h)

    def finite_horizon(self, H: int) -> GridValue:
        """V_H: H sweeps from V = 0 (exact H-step cost-to-go, no terminal cost)."""
        V = torch.zeros(len(self.S))
        for _ in range(H):
            V, _ = self.bellman(V)
        return GridValue(self.env, V.reshape(2, self.N, self.N), self.grid.h)


class OracleTerminal:
    """Exact grid value as an MPC terminal cost, one grid per problem in the batch.

    Call with x [B, ..., 3]; problem b uses ``values[b]`` (psi is ignored).
    """

    def __init__(self, values: list[GridValue]):
        self.ref = values[0]
        self.Vs = torch.stack([v.V.reshape(-1) for v in values])  # [B, 2*N*N]

    def __call__(self, x: Tensor, psi: Optional[Tensor] = None) -> Tensor:
        idx, wts = self.ref._corners(x)  # [B, ..., 4]
        B = x.shape[0]
        vals = torch.gather(self.Vs, 1, idx.reshape(B, -1)).reshape(idx.shape)
        return (vals * wts).sum(-1)


def solve_oracles(env: KeyDoor2D, reward: Reward, psi: Tensor, grid: GridSpec = GridSpec()) -> list[GridValue]:
    """One exact value per row of psi [B, P]."""
    return [KeyDoorDP(env, reward, p, grid).solve() for p in psi]


def greedy_rollout(env: KeyDoor2D, reward: Reward, psi: Tensor, V: GridValue,
                   x0: Tensor, max_steps: int, actions: Tensor) -> tuple[Tensor, Tensor]:
    """Roll out the one-step-lookahead policy argmin_a [cost + V(step(x, a))].

    With V = V_{H-1} this is the ideal H-step MPC without terminal cost.
    x0 [B, 3]; returns trajectories [B, T+1, 3] and success flags [B].
    """
    goal = KeyDoor2D.goal_from_psi(reward, psi)
    B, A = x0.shape[0], len(actions)
    psi_b = psi.expand(B, A, -1) if psi.ndim == 1 else psi[:, None].expand(B, A, -1)
    goal_b = goal if goal.ndim > 1 else goal.expand(B, 2)
    x, traj = x0, [x0]
    done = env.success(x, goal_b)
    for _ in range(max_steps):
        xa = x[:, None].expand(B, A, 3)
        ua = actions[None].expand(B, A, 2)
        xn = env.step(xa, ua)
        q = reward.stage_cost(env, xn, ua, psi_b) + V(xn)
        best = q.argmin(dim=1)
        x_next = xn[torch.arange(B), best]
        x = torch.where(done[:, None], x, x_next)
        done = done | env.success(x, goal_b)
        traj.append(x)
    return torch.stack(traj, 1), done
