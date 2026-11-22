"""Closed-loop sampling-MPC baselines on the key-door problem.

Controllers (configs/keydoor.yaml -> eval.controllers), all with the same
number of samples per control step:
  H5, H10        short horizon, no terminal cost   (expected: stuck at the door)
  H40, H60       long horizon, no terminal cost    (reference, slower)
  H*_oracleV     short horizon + exact DP value as terminal cost
                 (upper bound for what the learned V_theta can achieve)

Quantitative results use random starts and random goals; the figure uses the
main goal so paths are comparable.

Outputs in results/keydoor_baselines/: summary.csv, paths.png
Usage: python scripts/run_baselines.py [--config configs/keydoor.yaml] [--only short_H10,...]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Fallback: ~/Desktop is synced by iCloud, which can hide the venv's .pth files.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv

import matplotlib.pyplot as plt
import torch

from rloc.envs.keydoor2d_dp import GridSpec, OracleTerminal, solve_oracles
from rloc.eval.common import (ROOT, build_controller, load_config, make_keydoor, make_psi, maybe_mkdir,
                              run_episodes, seeded, time_per_step)
from rloc.eval.plots import draw_keydoor, plot_trajectories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/keydoor.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/keydoor_baselines"))
    ap.add_argument("--only", default=None, help="comma-separated controller names")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = maybe_mkdir(args.out)
    env, reward = make_keydoor(cfg)
    ev = cfg["eval"]
    grid = GridSpec(h=cfg["dp"]["grid_h"], actions_per_axis=cfg["dp"]["actions_per_axis"])

    gen = seeded(ev["seed"])
    x0 = env.sample_start_states(ev["num_episodes"], gen)
    goals = env.sample_goals(ev["num_episodes"], gen)
    psi = make_psi(cfg, reward, goals)

    # figure set: 12 starts, main goal
    x0_fig = x0[:12]
    psi_fig = make_psi(cfg, reward, torch.tensor(cfg["reward"]["main_goal"]).expand(12, 2))

    controllers = ev["controllers"]
    if args.only:
        controllers = {k: v for k, v in controllers.items() if k in args.only.split(",")}
    need_oracle = any(s["terminal"] == "oracle" for s in controllers.values())
    if need_oracle:
        print(f"solving exact DP for {len(psi)} goals ...")
        oracle_eval = OracleTerminal(solve_oracles(env, reward, psi, grid))
        oracle_fig = OracleTerminal(solve_oracles(env, reward, psi_fig[:1], grid) * 12)
        oracle_one = OracleTerminal(solve_oracles(env, reward, psi[:1], grid))

    rows, fig_data = [], {}
    for name, spec in controllers.items():
        oracle = spec["terminal"] == "oracle"
        term = lambda which: (which if oracle else None)  # noqa: E731

        ctrl = build_controller(env, reward, cfg, spec, term(oracle_eval if oracle else None))
        stats = run_episodes(env, reward, ctrl, x0, psi, ev["max_steps"])
        s = stats.summary()

        ctrl1 = build_controller(env, reward, cfg, spec, term(oracle_one if oracle else None))
        s["ms_per_step"] = 1e3 * time_per_step(ctrl1, env, x0, psi, ev["timing_steps"])

        ctrlf = build_controller(env, reward, cfg, spec, term(oracle_fig if oracle else None))
        fig_data[name] = run_episodes(env, reward, ctrlf, x0_fig, psi_fig, ev["max_steps"])

        rows.append({"controller": name, "horizon": spec["horizon"], "knots": spec["num_knots"] or spec["horizon"],
                     "terminal": spec["terminal"], **s})
        print(f"{name:>20}: success {s['success_rate']:.2f}  switch {s['switch_rate']:.2f}  "
              f"steps {s['mean_steps_success']:.1f}  cost {s['mean_cost']:.2f}  {s['ms_per_step']:.2f} ms/step")

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(fig_data)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, (name, st) in zip(axes, fig_data.items()):
        draw_keydoor(ax, env, cfg["reward"]["main_goal"])
        plot_trajectories(ax, st.traj, st.success)
        ax.set_title(f"{name}\n{st.success.float().mean():.0%} success", fontsize=10)
    fig.savefig(out / "paths.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
