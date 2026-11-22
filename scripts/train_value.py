"""Learn the key-door terminal value by fitted value iteration and evaluate it.

Training: eq. (2) of the proposal, targets from a T-step sampling MPC with the
previous value as terminal cost (rloc.core.fvi).  The goal is fixed to
reward.main_goal (value.goal_mode: fixed).

Evaluation:
  * error of V_theta against the exact grid value on held-out states,
  * closed-loop success of short-horizon MPC with no / learned / exact terminal value.

Outputs:
  checkpoints/value_keydoor_fixed.pt
  results/keydoor_value/{learning_curve.png, value_learned.png, value_error.png,
                         closed_loop.csv, paths.png, history.csv}

Usage: python scripts/train_value.py [--config configs/keydoor.yaml] [--eval-only]
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

from rloc.core.fvi import FittedValueIteration, FVIConfig
from rloc.core.mppi import MPPIConfig
from rloc.core.value_net import ValueNet
from rloc.envs.keydoor2d import KeyDoor2D
from rloc.envs.keydoor2d_dp import GridSpec, KeyDoorDP
from rloc.eval.common import (ROOT, build_controller, load_config, make_keydoor, make_psi, maybe_mkdir,
                              run_episodes, seeded, time_per_step)
from rloc.eval.plots import draw_keydoor, plot_trajectories, plot_value_heatmaps


def fvi_config(cfg: dict, iterations: int | None) -> FVIConfig:
    v, m = cfg["value"], cfg["mppi"]
    target = MPPIConfig(horizon=v["target_horizon"], num_samples=v["target_samples"], num_iters=v["target_iters"],
                        noise_sigma=m["noise_sigma"], temperature=m["temperature"], update=m["update"])
    return FVIConfig(iterations=iterations or v["iterations"], states_per_iter=v["states_per_iter"], target=target,
                     train_steps=v["train_steps"], batch_size=v["batch_size"], lr=v["lr"],
                     weight_decay=v["weight_decay"], anchor_weight=v["anchor_weight"],
                     anchor_states=v["anchor_states"], rollout_frac=v["rollout_frac"],
                     rollout_buffer=v["rollout_buffer"], rollout_starts=v["rollout_starts"],
                     rollout_steps=v["rollout_steps"], rollout_horizon=v["rollout_horizon"], seed=v["seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/keydoor.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/keydoor_value"))
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/value_keydoor_fixed.pt"))
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = maybe_mkdir(args.out)
    maybe_mkdir(Path(args.ckpt).parent)
    env, reward = make_keydoor(cfg)
    v, ev = cfg["value"], cfg["eval"]
    if v["goal_mode"] != "fixed":
        raise NotImplementedError("only goal_mode: fixed is implemented so far")

    goal = cfg["reward"]["main_goal"]
    psi_main = make_psi(cfg, reward, goal)
    exact = KeyDoorDP(env, reward, psi_main, GridSpec(cfg["dp"]["grid_h"], cfg["dp"]["actions_per_axis"])).solve()

    # held-out states for the value error
    x_eval = env.sample_states(v["eval_states"], seeded(v["seed"] + 100))
    v_true = exact(x_eval)
    k_eval = x_eval[:, 2] > 0.5

    def errors(net: ValueNet) -> dict:
        with torch.no_grad():
            err = (net(x_eval, psi_main) - v_true).abs()
        return {"mae": err.mean().item(), "rel_mae": (err.mean() / v_true.mean()).item(),
                "mae_k0": err[~k_eval].mean().item(), "mae_k1": err[k_eval].mean().item()}

    if args.eval_only:
        net = ValueNet.from_checkpoint(torch.load(args.ckpt))
        history = None
    else:
        torch.manual_seed(v["seed"])
        net = ValueNet(env.nx, reward.psi_dim, v["hidden"], v["out_dim"])
        stage = lambda x, u, p: reward.stage_cost(env, x, u, p)  # noqa: E731

        def sample_psi(n, gen):
            return psi_main.expand(n, -1).clone()

        def goal_states(psi, gen):
            g = KeyDoor2D.goal_from_psi(reward, psi)
            k = (torch.rand(len(psi), 1, generator=gen) < 0.5).float()
            return torch.cat([g, k], -1)

        fvi = FittedValueIteration(env, stage, net, fvi_config(cfg, args.iterations),
                                   sample_states=env.sample_states, sample_psi=sample_psi, goal_states=goal_states,
                                   sample_starts=lambda n, g: env.sample_start_states(n, g))
        history = fvi.run(callback=lambda log, n: errors(n))
        torch.save({"state_dict": net.state_dict(), "net_config": net.config(), "goal": goal,
                    "config": cfg}, args.ckpt)
        with open(out / "history.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "target_mean", "train_loss", "seconds", *history[0].metrics])
            for h in history:
                w.writerow([h.iteration, h.target_mean, h.train_loss, h.seconds, *h.metrics.values()])

        it = [h.iteration for h in history]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(it, [h.metrics["mae"] for h in history], "o-", ms=3, label="all states")
        ax[0].plot(it, [h.metrics["mae_k0"] for h in history], "--", label="door closed (k=0)")
        ax[0].plot(it, [h.metrics["mae_k1"] for h in history], "--", label="door open (k=1)")
        ax[0].set_xlabel("value-iteration round")
        ax[0].set_ylabel("mean |V_theta - V_exact|")
        ax[0].set_yscale("log")
        ax[0].legend(fontsize=8)
        ax[1].plot(it, [h.target_mean for h in history], "o-", ms=3)
        ax[1].set_ylabel("mean Bellman target (training states)")
        ax[1].set_xlabel("value-iteration round")
        fig.savefig(out / "learning_curve.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    final = errors(net)
    print("value error vs exact:", {k: round(x, 4) for k, x in final.items()},
          f"(mean exact value {v_true.mean():.3f})")

    vmax = float(exact.V.max())
    with torch.no_grad():
        plot_value_heatmaps(str(out / "value_learned.png"), env, lambda x: net(x, psi_main), goal,
                            "Learned value V_theta", vmax=vmax)
        plot_value_heatmaps(str(out / "value_error.png"), env, lambda x: (net(x, psi_main) - exact(x)).abs(), goal,
                            "|V_theta - V_exact|", cmap="magma", label="absolute error")

    # ------------------------------------------------------------------ #
    # closed loop: short-horizon MPC with no / learned / exact terminal value
    # ------------------------------------------------------------------ #
    n = ev["num_episodes"]
    x0 = env.sample_start_states(n, seeded(ev["seed"]))
    psi = psi_main.expand(n, -1)
    learned = lambda x, p: net(x, p)  # noqa: E731
    oracle = lambda x, p: exact(x)  # noqa: E731
    rows, paths = [], {}
    for H in v["eval_horizons"]:
        for name, term in [("none", None), ("learned", learned), ("exact", oracle)]:
            spec = {"horizon": H}
            st = run_episodes(env, reward, build_controller(env, reward, cfg, spec, term), x0, psi, ev["max_steps"])
            s = st.summary()
            s["ms_per_step"] = 1e3 * time_per_step(build_controller(env, reward, cfg, spec, term), env, x0, psi,
                                                    ev["timing_steps"])
            rows.append({"horizon": H, "terminal": name, **s})
            paths[(H, name)] = st
            print(f"H={H:3d} terminal={name:8s} success {s['success_rate']:.2f}  steps {s['mean_steps_success']:.1f}"
                  f"  cost {s['mean_cost']:.2f}  {s['ms_per_step']:.2f} ms/step")
    with open(out / "closed_loop.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    Hs, names = v["eval_horizons"], ["none", "learned", "exact"]
    fig, axes = plt.subplots(len(names), len(Hs), figsize=(4.2 * len(Hs), 4.3 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        for j, H in enumerate(Hs):
            st = paths[(H, name)]
            ax = axes[i, j]
            draw_keydoor(ax, env, goal)
            plot_trajectories(ax, st.traj[:12], st.success[:12])
            ax.set_title(f"H={H}, terminal: {name}\n{st.success.float().mean():.0%} success (all {n})", fontsize=10)
    fig.savefig(out / "paths.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
