"""Fitted value iteration with a sampling MPC as the Bellman operator.

Eq. (2) of the proposal, after Jordana et al. (2025) with MPPI instead of SQP.
Starting from V_0 = 0, each iteration k

1. samples states x_i and reward parameters psi_i,
2. computes targets  y_i = min_u  sum_{t<T} l_psi(x_t, u_t) + V_k(x_T; psi_i)
   with a T-step sampling MPC started from x_i (``MPPI.optimize(...).best_cost``),
3. fits V_{k+1} by regression on (x_i, psi_i) -> y_i, plus an anchor term
   alpha * V(x_goal; psi)^2 that pins the value to zero at goal states.

The network is warm-started from V_k between iterations; targets always use a
frozen copy of V_k.  Everything environment-specific comes in as callables, so
the same loop will be used for the arm tasks.

On-policy states (Jordana et al. add MPC rollout states too): a fraction
``rollout_frac`` of each iteration's states comes from closed-loop runs of the
short-horizon MPC with terminal cost V_k.  These states concentrate where the
current controller gets stuck, which is exactly where V_k is still wrong
(for the toy problem: in front of the closed door).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from rloc.core.mppi import MPPI, MPCController, MPPIConfig
from rloc.core.value_net import ValueNet

Tensor = torch.Tensor
Sampler = Callable[[int, torch.Generator], Tensor]


@dataclass
class FVIConfig:
    iterations: int = 30
    states_per_iter: int = 4096
    target: MPPIConfig = field(default_factory=lambda: MPPIConfig(horizon=5, num_samples=256, num_iters=3,
                                                                   noise_sigma=1.0, update="best"))
    target_chunk: int = 4096  # start states per MPPI call (memory bound)
    train_steps: int = 500  # gradient steps per iteration
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    anchor_weight: float = 1.0
    anchor_states: int = 256
    # on-policy states from closed-loop MPC with V_k
    rollout_frac: float = 0.0
    rollout_starts: int = 256
    rollout_steps: int = 40
    rollout_horizon: int = 5
    # Replay buffer of on-policy *states* (targets are recomputed with the
    # current V_k every round).  Without it, once the controller succeeds the
    # rollouts stop visiting the hard states, V forgets them and the
    # controller fails again.  0 = use only this round's rollouts.
    rollout_buffer: int = 0
    seed: int = 0


@dataclass
class IterationLog:
    iteration: int
    target_mean: float
    target_max: float
    train_loss: float
    seconds: float
    metrics: dict = field(default_factory=dict)


class FittedValueIteration:
    def __init__(
        self,
        env,
        stage_cost: Callable[[Tensor, Tensor, Tensor], Tensor],
        net: ValueNet,
        cfg: FVIConfig,
        sample_states: Sampler,
        sample_psi: Sampler,
        goal_states: Optional[Callable[[Tensor, torch.Generator], Tensor]] = None,
        sample_starts: Optional[Sampler] = None,
    ):
        """
        stage_cost(x', u, psi) -> cost; sample_states(n, gen) -> [n, nx];
        sample_psi(n, gen) -> [n, P]; goal_states(psi [n, P], gen) -> [n, nx]
        (zero-cost states used for the anchor term, optional);
        sample_starts(n, gen) -> [n, nx] start states of the on-policy rollouts
        (defaults to sample_states).
        """
        self.env, self.stage_cost, self.net, self.cfg = env, stage_cost, net, cfg
        self.sample_states, self.sample_psi, self.goal_states = sample_states, sample_psi, goal_states
        self.sample_starts = sample_starts or sample_states
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.history: list[IterationLog] = []
        self.frozen: Optional[ValueNet] = None  # V_k used for targets; None means V_0 = 0
        self._buf_x: Optional[Tensor] = None  # replay buffer of on-policy states
        self._buf_p: Optional[Tensor] = None

    # ------------------------------------------------------------------ #
    def terminal_fn(self, net: Optional[ValueNet]):
        if net is None:
            return None
        return lambda x, psi: net(x, psi)

    @torch.no_grad()
    def compute_targets(self, x: Tensor, psi: Tensor) -> Tensor:
        mppi = MPPI(self.env.step, self.stage_cost, self.env.u_low, self.env.u_high, self.cfg.target,
                    terminal_cost_fn=self.terminal_fn(self.frozen), generator=self.gen)
        ys = []
        for i in range(0, len(x), self.cfg.target_chunk):
            sl = slice(i, i + self.cfg.target_chunk)
            ys.append(mppi.optimize(x[sl], psi[sl]).best_cost)
        return torch.cat(ys)

    @torch.no_grad()
    def rollout_states(self, n: int) -> tuple[Tensor, Tensor]:
        """n states (and their psi) visited by the closed-loop MPC with terminal V_k."""
        cfg = self.cfg
        x = self.sample_starts(cfg.rollout_starts, self.gen)
        psi = self.sample_psi(cfg.rollout_starts, self.gen)
        mcfg = copy.copy(cfg.target)
        mcfg.horizon, mcfg.num_iters, mcfg.num_knots = cfg.rollout_horizon, 1, None
        ctrl = MPCController(MPPI(self.env.step, self.stage_cost, self.env.u_low, self.env.u_high, mcfg,
                                  terminal_cost_fn=self.terminal_fn(self.frozen), generator=self.gen))
        visited = []
        for _ in range(cfg.rollout_steps):
            x = self.env.step(x, ctrl.act(x, psi))
            visited.append(x)
        X = torch.stack(visited, 1).reshape(-1, x.shape[-1])
        P = psi[:, None].expand(-1, cfg.rollout_steps, -1).reshape(-1, psi.shape[-1])
        idx = torch.randint(0, len(X), (n,), generator=self.gen)
        return X[idx], P[idx]

    def fit(self, x: Tensor, psi: Tensor, y: Tensor) -> float:
        cfg, n = self.cfg, len(x)
        self.net.train()
        running = 0.0
        for step in range(cfg.train_steps):
            idx = torch.randint(0, n, (min(cfg.batch_size, n),), generator=self.gen)
            loss = ((self.net(x[idx], psi[idx]) - y[idx]) ** 2).mean()
            if self.goal_states is not None and cfg.anchor_weight > 0:
                psi_a = self.sample_psi(cfg.anchor_states, self.gen)
                xa = self.goal_states(psi_a, self.gen)
                loss = loss + cfg.anchor_weight * (self.net(xa, psi_a) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            running = 0.9 * running + 0.1 * loss.item() if step else loss.item()
        self.net.eval()
        return running

    def step(self) -> IterationLog:
        t0 = time.perf_counter()
        cfg = self.cfg
        n_roll = int(round(cfg.rollout_frac * cfg.states_per_iter))
        x = self.sample_states(cfg.states_per_iter - n_roll, self.gen)
        psi = self.sample_psi(cfg.states_per_iter - n_roll, self.gen)
        if not self.history:
            self.net.set_normalization(x, psi)
        if n_roll:
            xr, pr = self.rollout_states(n_roll)
            if cfg.rollout_buffer:
                if self._buf_x is None:
                    self._buf_x, self._buf_p = xr[:0], pr[:0]
                self._buf_x = torch.cat([self._buf_x, xr])[-cfg.rollout_buffer:]
                self._buf_p = torch.cat([self._buf_p, pr])[-cfg.rollout_buffer:]
                idx = torch.randint(0, len(self._buf_x), (n_roll,), generator=self.gen)
                xr, pr = self._buf_x[idx], self._buf_p[idx]
            x, psi = torch.cat([x, xr]), torch.cat([psi, pr])
        y = self.compute_targets(x, psi)
        loss = self.fit(x, psi, y)
        self.frozen = copy.deepcopy(self.net).eval()
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        log = IterationLog(len(self.history) + 1, y.mean().item(), y.max().item(), loss,
                           time.perf_counter() - t0)
        self.history.append(log)
        return log

    def run(self, callback: Optional[Callable[[IterationLog, ValueNet], dict]] = None,
            verbose: bool = True) -> list[IterationLog]:
        for _ in range(self.cfg.iterations):
            log = self.step()
            if callback is not None:
                log.metrics = callback(log, self.net) or {}
            if verbose:
                extra = "  ".join(f"{k} {v:.4f}" for k, v in log.metrics.items())
                print(f"iter {log.iteration:3d}  target mean {log.target_mean:.3f} max {log.target_max:.3f}  "
                      f"loss {log.train_loss:.4f}  {log.seconds:.1f}s  {extra}", flush=True)
        return self.history
