"""Plotting helpers for the key-door toy problem."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rloc.envs.keydoor2d import KeyDoor2D  # noqa: E402


def draw_keydoor(ax, env: KeyDoor2D, goal: Optional[Sequence[float]] = None, door_open: bool = False):
    c = env.cfg
    L, w, d = c.half_size, c.wall_x, c.door_half_width
    ax.plot([-L, L, L, -L, -L], [-L, -L, L, L, -L], color="0.3", lw=1)
    ax.plot([w, w], [-L, -d], color="k", lw=3)
    ax.plot([w, w], [d, L], color="k", lw=3)
    ax.plot([w, w], [-d, d], color="tab:green" if door_open else "tab:red",
            lw=3, ls=":" if door_open else "-")
    ax.add_patch(Circle(c.switch_pos, c.switch_radius, color="gold", ec="k", zorder=3))
    ax.annotate("switch", c.switch_pos, xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8)
    if goal is not None:
        g = np.asarray(goal, dtype=float)
        ax.plot(*g, marker="*", ms=14, color="tab:blue", mec="k", zorder=4)
    ax.set_xlim(-L - 0.05, L + 0.05)
    ax.set_ylim(-L - 0.05, L + 0.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_trajectories(ax, traj: torch.Tensor, success: Optional[torch.Tensor] = None):
    """traj [B, T+1, 3]; colour: before switch (grey) / after switch (blue)."""
    traj = traj.detach().cpu().numpy()
    for b in range(traj.shape[0]):
        p, k = traj[b, :, :2], traj[b, :, 2]
        pre = k < 0.5
        ax.plot(p[pre, 0], p[pre, 1], color="0.5", lw=1, alpha=0.8)
        if (~pre).any():
            i0 = max(np.argmax(~pre) - 1, 0)
            ax.plot(p[i0:, 0], p[i0:, 1], color="tab:blue", lw=1, alpha=0.8)
        ok = success is None or bool(success[b])
        ax.plot(*p[0], "o", ms=3, color="k")
        ax.plot(*p[-1], "x" if not ok else ".", ms=6 if not ok else 4, color="tab:red" if not ok else "tab:blue")


def plot_value_heatmaps(fig_path: str, env: KeyDoor2D, value_fn, goal, title: str = "", n: int = 150,
                        vmax: Optional[float] = None, cmap: str = "viridis", label: str = "cost-to-go"):
    """Two panels: value_fn(p, k=0) and value_fn(p, k=1)."""
    L = env.cfg.half_size
    xs = torch.linspace(-L + 1e-3, L - 1e-3, n)
    X, Y = torch.meshgrid(xs, xs, indexing="xy")
    p = torch.stack([X.reshape(-1), Y.reshape(-1)], -1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    vals = []
    for k in (0, 1):
        x = torch.cat([p, torch.full((len(p), 1), float(k))], -1)
        with torch.no_grad():
            vals.append(value_fn(x).reshape(n, n).cpu().numpy())
    vmax = vmax or max(v.max() for v in vals)
    for k, ax in enumerate(axes):
        im = ax.imshow(vals[k], origin="lower", extent=[-L, L, -L, L], cmap=cmap, vmin=0, vmax=vmax)
        draw_keydoor(ax, env, goal, door_open=bool(k))
        ax.set_title(f"k={k}" + (" — door open" if k else " — door closed"))
    fig.colorbar(im, ax=axes, shrink=0.85, label=label)
    if title:
        fig.suptitle(title)
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
