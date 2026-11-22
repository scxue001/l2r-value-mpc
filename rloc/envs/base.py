"""Interface shared by all environments (toy problem now, MuJoCo arm later).

An environment only has to provide batched dynamics and named entity
positions; the reward API builds costs on top of ``entity_pos``.  A MuJoCo
environment will implement ``step`` with ``mujoco.rollout`` behind the same
signature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

Tensor = torch.Tensor


class BatchedEnv(ABC):
    nx: int  # state dimension
    nu: int  # control dimension
    dt: float  # control period [s]
    u_low: Tensor  # [nu]
    u_high: Tensor  # [nu]

    @abstractmethod
    def step(self, x: Tensor, u: Tensor) -> Tensor:
        """x [..., nx], u [..., nu] -> next state [..., nx]."""

    @abstractmethod
    def entity_pos(self, x: Tensor, name: str) -> Tensor:
        """Position [..., d] of a named entity (e.g. "point", "hand", "block")."""

    @abstractmethod
    def sample_states(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        """States [n, nx] covering the region the value function must learn."""
