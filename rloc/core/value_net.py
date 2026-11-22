"""Value network V_theta(x; psi).

Following Jordana et al. (2025) the output is a sum of squares,

    V(x; psi) = 1/2 * || r_theta([x, psi]) ||^2,   r_theta: R^{nx+P} -> R^d,

so V >= 0 by construction and V = 0 is reachable (at the goal).  Hidden layers
use tanh.  Inputs are normalised with fixed statistics set once from data.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

Tensor = torch.Tensor


class ValueNet(nn.Module):
    def __init__(self, nx: int, psi_dim: int, hidden: Sequence[int] = (64, 64, 64), out_dim: int = 32):
        super().__init__()
        self.nx, self.psi_dim = nx, psi_dim
        layers, d = [], nx + psi_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.body = nn.Sequential(*layers)
        self.register_buffer("in_mean", torch.zeros(nx + psi_dim))
        self.register_buffer("in_std", torch.ones(nx + psi_dim))

    @torch.no_grad()
    def set_normalization(self, x: Tensor, psi: Tensor) -> None:
        """Fix input statistics from a representative batch (constant dims keep std 1)."""
        z = torch.cat([x, psi], -1).reshape(-1, self.nx + self.psi_dim)
        std = z.std(0)
        self.in_mean.copy_(z.mean(0))
        self.in_std.copy_(torch.where(std > 1e-6, std, torch.ones_like(std)))

    def forward(self, x: Tensor, psi: Tensor) -> Tensor:
        """x [..., nx], psi [..., P] (broadcastable) -> V [...]."""
        psi = psi.expand(*x.shape[:-1], self.psi_dim)
        z = (torch.cat([x, psi], -1) - self.in_mean) / self.in_std
        r = self.body(z)
        return 0.5 * (r * r).sum(-1)

    def config(self) -> dict:
        lin = [m for m in self.body if isinstance(m, nn.Linear)]
        return {"nx": self.nx, "psi_dim": self.psi_dim,
                "hidden": [m.out_features for m in lin[:-1]], "out_dim": lin[-1].out_features}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "ValueNet":
        net = cls(**ckpt["net_config"])
        net.load_state_dict(ckpt["state_dict"])
        return net.eval()
