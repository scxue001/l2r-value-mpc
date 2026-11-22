"""Exact value of the key-door problem and the ideal finite-horizon study.

Outputs (in results/keydoor_dp/):
  value_exact.png        V(p, k=0) and V(p, k=1) for the main goal
  value_H10.png          10-step cost-to-go without terminal cost (what short MPC "sees")
  horizon_study.png/csv  success of the ideal H-step lookahead policy vs H
  greedy_paths.png       paths of the optimal policy and of the ideal H=10 policy
  value_exact.pt         grid value, reused as the oracle terminal cost / ground truth

Usage: python scripts/exact_dp_keydoor.py [--config configs/keydoor.yaml]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Fallback: ~/Desktop is synced by iCloud, which can hide the venv's .pth files.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import time

import matplotlib.pyplot as plt
import torch

from rloc.envs.keydoor2d_dp import GridSpec, KeyDoorDP, greedy_rollout
from rloc.eval.common import ROOT, load_config, make_keydoor, make_psi, maybe_mkdir, seeded
from rloc.eval.plots import draw_keydoor, plot_trajectories, plot_value_heatmaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/keydoor.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/keydoor_dp"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = maybe_mkdir(args.out)

    env, reward = make_keydoor(cfg)
    goal = cfg["reward"]["main_goal"]
    psi = make_psi(cfg, reward, goal)
    grid = GridSpec(h=cfg["dp"]["grid_h"], actions_per_axis=cfg["dp"]["actions_per_axis"])

    t0 = time.perf_counter()
    dp = KeyDoorDP(env, reward, psi, grid)
    V = dp.solve()
    print(f"value iteration: {dp.iterations} sweeps, {time.perf_counter() - t0:.2f} s, grid {dp.N}x{dp.N}x2")
    torch.save({"V": V.V, "h": grid.h, "goal": goal, "config": cfg}, out / "value_exact.pt")

    vmax = float(V.V.max())
    plot_value_heatmaps(str(out / "value_exact.png"), env, V, goal, "Exact infinite-horizon value", vmax=vmax)
    plot_value_heatmaps(str(out / "value_H10.png"), env, dp.finite_horizon(10), goal,
                        "10-step cost-to-go, no terminal cost", vmax=vmax)

    # ideal finite-horizon MPC: greedy w.r.t. V_{H-1} (exact H-step lookahead)
    ev = cfg["eval"]
    x0 = env.sample_start_states(ev["num_episodes"], seeded(ev["seed"]))
    rows = []
    _, ok = greedy_rollout(env, reward, psi, V, x0, ev["max_steps"], dp.actions)
    rows.append(("inf", ok.float().mean().item()))
    for H in cfg["dp"]["horizons"]:
        traj, ok = greedy_rollout(env, reward, psi, dp.finite_horizon(H - 1), x0, ev["max_steps"], dp.actions)
        rows.append((H, ok.float().mean().item()))
    print("ideal H-step lookahead (no terminal cost), success rate:")
    for H, s in rows:
        print(f"  H = {H!s:>4}: {s:.2f}")
    with open(out / "horizon_study.csv", "w", newline="") as f:
        csv.writer(f).writerows([("horizon", "success_rate"), *rows])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    hs = [r[0] for r in rows[1:]]
    ax.plot(hs, [r[1] for r in rows[1:]], "o-", label="exact H-step lookahead, no terminal cost")
    ax.axhline(rows[0][1], color="k", ls="--", label="infinite horizon (exact V)")
    ax.set_xlabel("horizon H (steps of %.2f s)" % env.dt)
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    fig.savefig(out / "horizon_study.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # paths: optimal vs ideal H = 10
    sub = x0[:12]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (name, Vf) in zip(axes, [("optimal (exact V)", V), ("ideal H=10, no terminal cost", dp.finite_horizon(9))]):
        traj, ok = greedy_rollout(env, reward, psi, Vf, sub, ev["max_steps"], dp.actions)
        draw_keydoor(ax, env, goal)
        plot_trajectories(ax, traj, ok)
        ax.set_title(f"{name}: {ok.float().mean():.0%} success")
    fig.savefig(out / "greedy_paths.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
