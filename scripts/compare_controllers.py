"""Controller comparison and horizon sweep on the key-door problem.

The controllers from the proposal, all with the same samples per control step:
  reward only      the task reward alone (L2R baseline)
  static shaping   + w * |p - switch|^2      (expressible as an L2R Distance term)
  staged shaping   + w * (1 - k) |p - switch|^2  (hand-written, needs the task stage)
  learned V        + goal-conditioned learned terminal value (ours)
  exact V          + exact grid value (upper bound)
The long-horizon reference is the "reward only" controller at large H.

Shaping weights are tuned at compare.tune_horizon on validation goals that are
disjoint from the test goals.  Test goals are the value-training script's
"unseen goals" set (never used to train the value).

Outputs (results/keydoor_compare/): tuning.csv, horizon_sweep.csv,
horizon_sweep.png, paths.png

Usage: python scripts/compare_controllers.py [--config configs/keydoor.yaml] [--horizons 3,5,10]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Fallback: ~/Desktop is synced by iCloud, which can hide the venv's .pth files.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import math

import matplotlib.pyplot as plt
import torch

from rloc.core.value_net import ValueNet
from rloc.envs.keydoor2d import KeyDoor2D
from rloc.envs.keydoor2d_dp import GridSpec, OracleTerminal, solve_oracles
from rloc.eval.common import (ROOT, build_controller, load_config, make_keydoor, make_psi_sampler, make_test_sets,
                              maybe_mkdir, run_episodes, seeded, time_per_step)
from rloc.eval.plots import draw_keydoor, plot_trajectories

CONTROLLERS = ["reward only", "static shaping", "staged shaping", "learned V", "exact V"]
COLORS = {"reward only": "tab:gray", "static shaping": "tab:orange", "staged shaping": "tab:green",
          "learned V": "tab:blue", "exact V": "k"}


def knots_for(H: int, spacing: int):
    return None if H <= 10 else math.ceil(H / spacing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/keydoor.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/keydoor_compare"))
    ap.add_argument("--horizons", default=None, help="comma-separated, overrides compare.horizons")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cc, ev = cfg["compare"], cfg["eval"]
    out = maybe_mkdir(args.out)
    env, reward = make_keydoor(cfg)
    horizons = [int(h) for h in args.horizons.split(",")] if args.horizons else cc["horizons"]

    ckpt = Path(cc["value_ckpt"])
    ckpt = ckpt if ckpt.is_absolute() else ROOT / ckpt
    if not ckpt.exists():
        sys.exit(f"{ckpt} not found: train it with  python scripts/train_value.py --goal-mode random")
    net = ValueNet.from_checkpoint(torch.load(ckpt))
    learned = lambda x, p: net(x, p)  # noqa: E731
    grid = GridSpec(cfg["dp"]["grid_h"], cfg["dp"]["actions_per_axis"])
    seed = cfg["value"]["seed"]

    def controller(name: str, H: int, weights: dict, oracle=None):
        spec = {"horizon": H, "num_knots": knots_for(H, cc["knot_spacing"])}
        if name == "static shaping":
            return build_controller(env, reward, cfg, spec,
                                    extra_cost=lambda x, u, p: env.switch_shaping(x, weights["static"], False))
        if name == "staged shaping":
            return build_controller(env, reward, cfg, spec,
                                    extra_cost=lambda x, u, p: env.switch_shaping(x, weights["staged"], True))
        if name == "learned V":
            return build_controller(env, reward, cfg, spec, terminal=learned)
        if name == "exact V":
            return build_controller(env, reward, cfg, spec, terminal=oracle)
        return build_controller(env, reward, cfg, spec)

    # ------------------------------------------------------------------ #
    # 1. tune shaping weights on validation goals (disjoint from the test goals)
    # ------------------------------------------------------------------ #
    rg = cfg["value"]["random_goals"]
    tune_psi_set = make_psi_sampler(cfg, env, reward, "random", exclude=rg["holdout_box"])(
        cc["tune_goals"], seeded(seed + 400))
    nt = cc["tune_episodes"]
    tune_psi = tune_psi_set[torch.arange(nt) % len(tune_psi_set)]
    tune_x0 = env.sample_start_states(nt, seeded(seed + 400))
    tune_rows, best = [], {}
    for kind, staged in (("static", False), ("staged", True)):
        scores = []
        for w in cc["shaping_weights"]:
            st = run_episodes(env, reward, controller(f"{kind} shaping", cc["tune_horizon"], {kind: w}),
                              tune_x0, tune_psi, ev["max_steps"])
            s = st.summary()
            tune_rows.append({"shaping": kind, "weight": w, **s})
            scores.append((s["success_rate"], -s["mean_cost"], w))
            print(f"tune {kind:6s} w={w:<5}: success {s['success_rate']:.2f}  switch {s['switch_rate']:.2f}  "
                  f"cost {s['mean_cost']:.2f}", flush=True)
        best[kind] = max(scores)[2]
    print(f"chosen shaping weights: {best}")
    with open(out / "tuning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tune_rows[0]))
        w.writeheader()
        w.writerows(tune_rows)

    # ------------------------------------------------------------------ #
    # 2. horizon sweep on the test goals
    # ------------------------------------------------------------------ #
    psis = make_test_sets(cfg, env, reward, "random", seed)[cc["test_set"]]
    grids = solve_oracles(env, reward, psis, grid)
    n = ev["num_episodes"]
    idx = torch.arange(n) % len(psis)
    psi = psis[idx]
    x0 = env.sample_start_states(n, seeded(ev["seed"]))
    oracle = OracleTerminal([grids[i] for i in idx.tolist()])
    oracle_one = OracleTerminal([grids[0]])

    rows, paths = [], {}
    for H in horizons:
        for name in CONTROLLERS:
            st = run_episodes(env, reward, controller(name, H, best, oracle), x0, psi, ev["max_steps"])
            s = st.summary()
            s["stages"] = s["switch_rate"] + s["success_rate"]
            s["ms_per_step"] = 1e3 * time_per_step(controller(name, H, best, oracle_one), env, x0, psi,
                                                    ev["timing_steps"])
            rows.append({"controller": name, "horizon": H, "knots": knots_for(H, cc["knot_spacing"]) or H, **s})
            if H == cc["paths_horizon"]:
                paths[name] = st
            print(f"H={H:3d} {name:15s} success {s['success_rate']:.2f}  stages {s['stages']:.2f}  "
                  f"steps {s['mean_steps_success']:.1f}  cost {s['mean_cost']:.2f}  {s['ms_per_step']:.2f} ms/step",
                  flush=True)
    with open(out / "horizon_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("shortest horizon with success >= 0.9:")
    for name in CONTROLLERS:
        ok = [r["horizon"] for r in rows if r["controller"] == name and r["success_rate"] >= 0.9]
        print(f"  {name:15s} {min(ok) if ok else 'none'}")

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name in CONTROLLERS:
        r = [x for x in rows if x["controller"] == name]
        hs = [x["horizon"] for x in r]
        kw = dict(color=COLORS[name], marker="o", ms=4, label=name, ls="--" if name == "exact V" else "-")
        axes[0].plot(hs, [x["success_rate"] for x in r], **kw)
        axes[1].plot(hs, [x["stages"] for x in r], **kw)
        axes[2].plot(hs, [x["ms_per_step"] for x in r], **kw)
    for ax, lab in zip(axes, ["success rate", "stages completed (switch + goal, max 2)", "ms per control step"]):
        ax.set_xscale("log")
        ax.set_xticks(horizons)
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_xlabel("horizon H (steps of %.1f s)" % env.dt)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 2.05)
    axes[0].legend(fontsize=8)
    fig.savefig(out / "horizon_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    if paths:
        goals = KeyDoor2D.goal_from_psi(reward, psi)
        fig, axes = plt.subplots(1, len(paths), figsize=(4.2 * len(paths), 4.4), squeeze=False)
        for ax, (name, st) in zip(axes[0], paths.items()):
            draw_keydoor(ax, env)
            plot_trajectories(ax, st.traj[:12], st.success[:12], goals[:12])
            ax.set_title(f"{name}, H={cc['paths_horizon']}\n{st.success.float().mean():.0%} success (all {n})",
                         fontsize=10)
        fig.savefig(out / "paths.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
