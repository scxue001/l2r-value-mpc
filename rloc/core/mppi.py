"""Batched MPPI (Williams et al., 2016) in PyTorch.

The solver optimises ``B`` independent problems at once, each with ``K``
sampled control sequences.  The same code is used

* as the real-time controller (``B`` = number of parallel episodes, often 1),
* as the target generator for fitted value iteration (``B`` = thousands of
  sampled start states, eq. (2) of the proposal).

Controls are parameterised by ``num_knots`` knots that are linearly
interpolated over the horizon (as in MJPC's sampling planners).  With
``num_knots = horizon`` (the default) this is ordinary per-step MPPI.

Cost of a control sequence for problem b::

    J = sum_{t=0}^{H-1} stage_cost(x_{t+1}, u_t; psi_b) + terminal_cost(x_H; psi_b)

Sample weights use a normalised cost so the temperature does not depend on
the cost scale: ``w_k ∝ exp(-(J_k - J_min) / ((J_max - J_min) * temperature))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

Tensor = torch.Tensor
StepFn = Callable[[Tensor, Tensor], Tensor]  # (x [...,nx], u [...,nu]) -> x'
StageCostFn = Callable[[Tensor, Tensor, Tensor], Tensor]  # (x', u, psi) -> [...]
TerminalCostFn = Callable[[Tensor, Tensor], Tensor]  # (x, psi) -> [...]


@dataclass
class MPPIConfig:
    horizon: int = 10
    num_samples: int = 256
    num_iters: int = 1  # optimisation iterations per call
    noise_sigma: float = 0.5  # std of knot noise, in control units
    temperature: float = 0.1  # relative to the normalised cost range
    num_knots: Optional[int] = None  # None -> one knot per step
    keep_nominal: bool = True  # sample 0 is the noise-free nominal sequence
    # "mppi": exponentially weighted average of the samples (Williams et al.)
    # "best": keep the lowest-cost sample (MJPC Predictive Sampling, used by L2R)
    update: str = "mppi"


@dataclass
class MPPIResult:
    theta: Tensor  # [B, n_knots, nu] updated knots
    controls: Tensor  # [B, H, nu] controls of the updated knots
    cost: Tensor  # [B] cost of the updated (mean) sequence
    best_cost: Tensor  # [B] lowest cost seen (samples of all iterations + mean)
    best_controls: Tensor  # [B, H, nu] controls achieving best_cost
    states: Tensor  # [B, H+1, nx] rollout of the updated sequence


def linear_interp_matrix(query: Tensor, knots: Tensor) -> Tensor:
    """Matrix M [len(query), len(knots)] with (M @ values) = interp(query).

    Queries outside the knot range are clamped (zero-order hold at the ends).
    """
    n = knots.numel()
    if n == 1:
        return torch.ones(query.numel(), 1, dtype=knots.dtype)
    q = query.clamp(knots[0], knots[-1])
    idx = torch.searchsorted(knots, q, right=True).clamp(1, n - 1)
    lo, hi = knots[idx - 1], knots[idx]
    frac = (q - lo) / (hi - lo)
    M = torch.zeros(q.numel(), n, dtype=knots.dtype)
    rows = torch.arange(q.numel())
    M[rows, idx - 1] = 1.0 - frac
    M[rows, idx] += frac
    return M


class MPPI:
    def __init__(
        self,
        step_fn: StepFn,
        stage_cost_fn: StageCostFn,
        u_low: Tensor,
        u_high: Tensor,
        cfg: MPPIConfig,
        terminal_cost_fn: Optional[TerminalCostFn] = None,
        generator: Optional[torch.Generator] = None,
    ):
        self.step_fn = step_fn
        self.stage_cost_fn = stage_cost_fn
        self.terminal_cost_fn = terminal_cost_fn
        self.cfg = cfg
        self.u_low = torch.as_tensor(u_low, dtype=torch.float32)
        self.u_high = torch.as_tensor(u_high, dtype=torch.float32)
        self.nu = self.u_low.numel()
        self.gen = generator

        H = cfg.horizon
        self.n_knots = cfg.num_knots or H
        if not 1 <= self.n_knots <= H:
            raise ValueError("num_knots must be in [1, horizon]")
        steps = torch.arange(H, dtype=torch.float32)
        knot_t = torch.linspace(0, H - 1, self.n_knots) if self.n_knots > 1 else torch.zeros(1)
        self.interp = linear_interp_matrix(steps, knot_t)  # [H, n_knots]
        self.shift_mat = linear_interp_matrix(knot_t + 1.0, knot_t)  # [n_knots, n_knots]

    # ------------------------------------------------------------------ #
    def init_theta(self, batch: int) -> Tensor:
        """Nominal knots: the middle of the control box (zero for symmetric boxes)."""
        mid = 0.5 * (self.u_low + self.u_high)
        return mid.expand(batch, self.n_knots, self.nu).clone()

    def knots_to_controls(self, theta: Tensor) -> Tensor:
        """[..., n_knots, nu] -> [..., H, nu]."""
        return torch.einsum("hk,...ku->...hu", self.interp, theta)

    def shift(self, theta: Tensor) -> Tensor:
        """Advance the knots by one control step (receding-horizon warm start)."""
        return torch.einsum("jk,...ku->...ju", self.shift_mat, theta)

    def rollout(self, x0: Tensor, U: Tensor, psi: Tensor) -> tuple[Tensor, Tensor]:
        """Roll out controls U [B, K, H, nu] from x0 [B, nx].

        Returns states [B, K, H+1, nx] and total cost [B, K].
        """
        B, K, H, _ = U.shape
        x = x0[:, None, :].expand(B, K, x0.shape[-1])
        psi_b = psi[:, None, :].expand(B, K, psi.shape[-1])
        states = [x]
        cost = torch.zeros(B, K, dtype=x0.dtype)
        for t in range(H):
            u = U[:, :, t]
            x = self.step_fn(x, u)
            cost = cost + self.stage_cost_fn(x, u, psi_b)
            states.append(x)
        if self.terminal_cost_fn is not None:
            cost = cost + self.terminal_cost_fn(x, psi_b)
        return torch.stack(states, dim=2), cost

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def optimize(
        self,
        x0: Tensor,
        psi: Tensor,
        theta: Optional[Tensor] = None,
        num_iters: Optional[int] = None,
    ) -> MPPIResult:
        """Run MPPI iterations from start states x0 [B, nx] with params psi [B, P]."""
        cfg = self.cfg
        if cfg.update not in ("mppi", "best"):
            raise ValueError(f"unknown update rule {cfg.update!r}")
        B = x0.shape[0]
        K = cfg.num_samples
        theta = self.init_theta(B) if theta is None else theta.clone()
        best_cost = torch.full((B,), float("inf"))
        best_theta = theta.clone()

        for _ in range(num_iters or cfg.num_iters):
            noise = torch.randn(B, K, self.n_knots, self.nu, generator=self.gen) * cfg.noise_sigma
            if cfg.keep_nominal:
                noise[:, 0] = 0.0
            samples = torch.max(torch.min(theta[:, None] + noise, self.u_high), self.u_low)
            _, cost = self.rollout(x0, self.knots_to_controls(samples), psi)

            # track the best sample (used as value-iteration target)
            c_min, k_min = cost.min(dim=1)
            improved = c_min < best_cost
            best_cost = torch.where(improved, c_min, best_cost)
            best_theta[improved] = samples[improved, k_min[improved]]

            if cfg.update == "best":
                theta = samples[torch.arange(B), k_min]
            else:  # normalised soft-min weights
                c_max = cost.max(dim=1, keepdim=True).values
                norm = (cost - c_min[:, None]) / (c_max - c_min[:, None] + 1e-9)
                w = torch.softmax(-norm / cfg.temperature, dim=1)  # [B, K]
                theta = torch.einsum("bk,bkju->bju", w, samples)

        U = self.knots_to_controls(theta)
        states, cost = self.rollout(x0, U[:, None], psi)
        cost, states = cost[:, 0], states[:, 0]
        improved = cost < best_cost
        best_cost = torch.where(improved, cost, best_cost)
        best_theta[improved] = theta[improved]
        return MPPIResult(
            theta=theta,
            controls=U,
            cost=cost,
            best_cost=best_cost,
            best_controls=self.knots_to_controls(best_theta),
            states=states,
        )


class MPCController:
    """Receding-horizon wrapper: optimise, apply u_0, shift, repeat."""

    def __init__(self, mppi: MPPI):
        self.mppi = mppi
        self.theta: Optional[Tensor] = None

    def reset(self) -> None:
        self.theta = None

    @torch.no_grad()
    def act(self, x: Tensor, psi: Tensor) -> Tensor:
        """x [B, nx], psi [B, P] -> u [B, nu]."""
        if self.theta is None or self.theta.shape[0] != x.shape[0]:
            self.theta = self.mppi.init_theta(x.shape[0])
        res = self.mppi.optimize(x, psi, self.theta)
        self.theta = self.mppi.shift(res.theta)
        return res.controls[:, 0]
