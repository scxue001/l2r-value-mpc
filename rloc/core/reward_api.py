"""L2R-style reward API.

Following Language to Rewards (Yu et al., 2023), a reward is a fixed list of
named residual terms.  Each term contributes

    weight * norm(residual(x, u; params))

to the stage cost.  The flat parameter vector ``psi`` concatenates, for every
term in order, ``[weight, *params]``.  ``psi`` is what the LLM writes (later in
the project) and what the learned value function is conditioned on.  A term
with weight 0 is disabled, exactly like L2R lets the LLM switch terms off.

Everything is batched: ``x``, ``u`` and ``psi`` may carry any leading
dimensions as long as they broadcast.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import torch

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Norms (MJPC-style).  All map a residual [..., d] to a scalar [...].
# --------------------------------------------------------------------------- #
def _quadratic(r: Tensor) -> Tensor:
    return (r * r).sum(-1)


def _smooth_l2(r: Tensor, p: float = 0.05) -> Tensor:
    # sqrt(|r|^2 + p^2) - p : quadratic near 0, linear far away.
    return torch.sqrt((r * r).sum(-1) + p * p) - p


NORMS: dict[str, Callable[[Tensor], Tensor]] = {
    "quadratic": _quadratic,
    "smooth_l2": _smooth_l2,
}


# --------------------------------------------------------------------------- #
# Terms
# --------------------------------------------------------------------------- #
class Term:
    """One residual term.  Subclasses declare their parameters and residual."""

    #: names and sizes of the parameters this term reads from psi
    param_spec: tuple[tuple[str, int], ...] = ()
    #: short description used later to build the LLM prompt
    doc: str = ""

    def __init__(self, norm: str = "quadratic"):
        if norm not in NORMS:
            raise ValueError(f"unknown norm {norm!r}; choose from {list(NORMS)}")
        self.norm = norm

    @property
    def param_dim(self) -> int:
        return sum(d for _, d in self.param_spec)

    def residual(self, env, x: Tensor, u: Tensor, params: Mapping[str, Tensor]) -> Tensor:
        raise NotImplementedError

    def cost(self, env, x: Tensor, u: Tensor, params: Mapping[str, Tensor]) -> Tensor:
        return NORMS[self.norm](self.residual(env, x, u, params))


class PositionTarget(Term):
    """Move an entity to a target position (L2R: set_obj_xy/z_position_reward)."""

    doc = "move `entity` to `target`"

    def __init__(self, entity: str, dim: int, norm: str = "quadratic"):
        super().__init__(norm)
        self.entity = entity
        self.param_spec = (("target", dim),)

    def residual(self, env, x, u, params):
        return env.entity_pos(x, self.entity) - params["target"]


class Distance(Term):
    """Bring two entities together (L2R: set_l2_distance_reward(a, b))."""

    doc = "bring `entity_a` close to `entity_b`"

    def __init__(self, entity_a: str, entity_b: str, norm: str = "quadratic"):
        super().__init__(norm)
        self.entity_a, self.entity_b = entity_a, entity_b

    def residual(self, env, x, u, params):
        return env.entity_pos(x, self.entity_a) - env.entity_pos(x, self.entity_b)


class ControlEffort(Term):
    """Penalise control magnitude."""

    doc = "penalise control effort"

    def residual(self, env, x, u, params):
        return u


# --------------------------------------------------------------------------- #
# Reward = ordered set of terms + psi layout
# --------------------------------------------------------------------------- #
class Reward:
    """An ordered set of named terms and the layout of the flat vector psi."""

    def __init__(self, terms: Mapping[str, Term]):
        self.terms: dict[str, Term] = dict(terms)
        self._slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, term in self.terms.items():
            size = 1 + term.param_dim  # weight + params
            self._slices[name] = (offset, offset + size)
            offset += size
        self.psi_dim = offset

    # -- psi <-> dict -------------------------------------------------------- #
    def pack(self, spec: Mapping[str, Mapping[str, object]], dtype=torch.float32) -> Tensor:
        """Build psi from ``{term: {"weight": w, <param>: value, ...}}``.

        Terms missing from ``spec`` get weight 0 (disabled) and zero params.
        """
        unknown = set(spec) - set(self.terms)
        if unknown:
            raise KeyError(f"unknown reward terms: {sorted(unknown)}")
        psi = torch.zeros(self.psi_dim, dtype=dtype)
        for name, term in self.terms.items():
            if name not in spec:
                continue
            lo, _ = self._slices[name]
            psi[lo] = float(spec[name].get("weight", 1.0))
            i = lo + 1
            for pname, pdim in term.param_spec:
                val = torch.as_tensor(spec[name][pname], dtype=dtype).reshape(pdim)
                psi[i : i + pdim] = val
                i += pdim
        return psi

    def pack_batch(self, n: int, spec: Mapping[str, Mapping[str, object]], dtype=torch.float32) -> Tensor:
        """Vectorised ``pack`` for n rows.

        Each value may be a scalar / [d] (shared by all rows) or [n] / [n, d]
        (one per row).  Terms missing from ``spec`` are disabled.
        """
        unknown = set(spec) - set(self.terms)
        if unknown:
            raise KeyError(f"unknown reward terms: {sorted(unknown)}")
        psi = torch.zeros(n, self.psi_dim, dtype=dtype)
        for name, term in self.terms.items():
            if name not in spec:
                continue
            lo, _ = self._slices[name]
            w = torch.as_tensor(spec[name].get("weight", 1.0), dtype=dtype)
            psi[:, lo] = w.expand(n) if w.ndim == 0 else w.reshape(n)
            i = lo + 1
            for pname, pdim in term.param_spec:
                val = torch.as_tensor(spec[name][pname], dtype=dtype)
                psi[:, i : i + pdim] = val.reshape(-1, pdim).expand(n, pdim)
                i += pdim
        return psi

    def unpack(self, psi: Tensor) -> dict[str, tuple[Tensor, dict[str, Tensor]]]:
        """Split psi [..., P] into ``{term: (weight [...], {param: [..., d]})}``."""
        out = {}
        for name, term in self.terms.items():
            lo, _ = self._slices[name]
            weight = psi[..., lo]
            params, i = {}, lo + 1
            for pname, pdim in term.param_spec:
                params[pname] = psi[..., i : i + pdim]
                i += pdim
            out[name] = (weight, params)
        return out

    # -- cost ---------------------------------------------------------------- #
    def stage_cost(self, env, x: Tensor, u: Tensor, psi: Tensor) -> Tensor:
        """Running cost of arriving in ``x`` after applying ``u``, times dt."""
        total = 0.0
        for name, (weight, params) in self.unpack(psi).items():
            total = total + weight * self.terms[name].cost(env, x, u, params)
        return total * env.dt

    def describe(self) -> str:
        """Human/LLM readable description of the API (used for prompts later)."""
        lines = []
        for name, term in self.terms.items():
            ps = ", ".join(f"{p}[{d}]" for p, d in term.param_spec) or "-"
            lines.append(f"{name}: {type(term).__name__}({term.doc}); params: weight, {ps}")
        return "\n".join(lines)

    def names(self) -> Sequence[str]:
        return list(self.terms)
