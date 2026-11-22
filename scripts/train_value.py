"""Learn the key-door terminal value by fitted value iteration and evaluate it.

Training: eq. (2) of the proposal, targets from a T-step sampling MPC with the
previous value as terminal cost (rloc.core.fvi).

Goal modes (value.goal_mode, or --goal-mode):
  fixed   psi is the single main goal (reward.main_goal).
  random  psi = (goal, goal weight, effort weight) is sampled for every state, so
          one network V(x; psi) serves every goal.  Goals inside
          value.random_goals.holdout_box are never used in training.

Evaluation, per test set (fixed: the main goal; random: unseen goals from the
training distribution, and goals inside the held-out box):
  * error of V_theta against the exact grid value (solved per test goal),
  * closed-loop success of short-horizon MPC with no / learned / exact terminal value.

Outputs (results/keydoor_value[_goals]/):
  learning_curve.png, history.csv, closed_loop.csv, value_error.csv,
  value_learned.png, value_error.png (first test goal), value_comparison.png,
  paths_<test set>.png;  checkpoint in checkpoints/value_keydoor_<mode>.pt

Usage: python scripts/train_value.py [--goal-mode fixed|random] [--eval-only]
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
from rloc.envs.keydoor2d_dp import GridSpec, OracleTerminal, solve_oracles
from rloc.eval.common import (ROOT, build_controller, load_config, make_keydoor, make_psi, maybe_mkdir,
                              run_episodes, seeded, time_per_step)
from rloc.eval.plots import (draw_keydoor, plot_trajectories, plot_value_comparison,
                             plot_value_heatmaps)


def value_settings(cfg: dict, mode: str) -> dict:
    """cfg["value"], with the random-goal overrides applied in random mode."""
    v = dict(cfg["value"])
    if mode == "random":
        rg = v["random_goals"]
        v.update({k: val for k, val in rg.items() if k in v})
    return v


def fvi_config(cfg: dict, v: dict, iterations: int | None) -> FVIConfig:
    m = cfg["mppi"]
    target = MPPIConfig(horizon=v["target_horizon"], num_samples=v["target_samples"], num_iters=v["target_iters"],
                        noise_sigma=m["noise_sigma"], temperature=m["temperature"], update=m["update"])
    return FVIConfig(iterations=iterations or v["iterations"], states_per_iter=v["states_per_iter"], target=target,
                     train_steps=v["train_steps"], batch_size=v["batch_size"], lr=v["lr"],
                     weight_decay=v["weight_decay"], anchor_weight=v["anchor_weight"],
                     anchor_states=v["anchor_states"], rollout_frac=v["rollout_frac"],
                     rollout_buffer=v["rollout_buffer"], rollout_starts=v["rollout_starts"],
                     rollout_steps=v["rollout_steps"], rollout_horizon=v["rollout_horizon"], seed=v["seed"])


def make_psi_sampler(cfg, env, reward, mode, region=None, exclude=None):
    """Callable (n, gen) -> psi [n, P]."""
    if mode == "fixed":
        psi_main = make_psi(cfg, reward, cfg["reward"]["main_goal"])
        return lambda n, g: psi_main.expand(n, -1).clone()
    rg = cfg["value"]["random_goals"]

    def sample(n, g):
        goals = env.sample_goals(n, g, region=region, exclude=exclude)
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g)  # noqa: E731
        return KeyDoor2D.make_psi(reward, goals, u(*rg["goal_weight_range"]), u(*rg["effort_weight_range"]))

    return sample


def test_sets(cfg, env, reward, mode, seed) -> dict[str, torch.Tensor]:
    if mode == "fixed":
        return {"main_goal": make_psi(cfg, reward, cfg["reward"]["main_goal"])[None]}
    rg = cfg["value"]["random_goals"]
    n = rg["test_goals"]
    unseen = make_psi_sampler(cfg, env, reward, mode, exclude=rg["holdout_box"])(n, seeded(seed + 200))
    held_out = make_psi_sampler(cfg, env, reward, mode, region=rg["holdout_box"])(n, seeded(seed + 300))
    return {"unseen_goals": unseen, "held_out_region": held_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/keydoor.yaml"))
    ap.add_argument("--goal-mode", choices=["fixed", "random"], default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mode = args.goal_mode or cfg["value"]["goal_mode"]
    v, ev = value_settings(cfg, mode), cfg["eval"]
    out = maybe_mkdir(args.out or ROOT / ("results/keydoor_value" if mode == "fixed" else "results/keydoor_value_goals"))
    ckpt = Path(args.ckpt or ROOT / f"checkpoints/value_keydoor_{mode}.pt")
    maybe_mkdir(ckpt.parent)
    env, reward = make_keydoor(cfg)
    grid = GridSpec(cfg["dp"]["grid_h"], cfg["dp"]["actions_per_axis"])

    # test goals and their exact values
    sets = test_sets(cfg, env, reward, mode, v["seed"])
    exact = {name: solve_oracles(env, reward, psis, grid) for name, psis in sets.items()}
    x_eval = env.sample_states(v["eval_states"], seeded(v["seed"] + 100))
    k_eval = x_eval[:, 2] > 0.5

    def errors(net: ValueNet, name: str, max_goals: int | None = None) -> dict:
        psis, grids = sets[name][:max_goals], exact[name][:max_goals]
        err, tot = [], []
        with torch.no_grad():
            for psi, V in zip(psis, grids):
                vt = V(x_eval)
                err.append((net(x_eval, psi) - vt).abs())
                tot.append(vt)
        err, tot = torch.stack(err), torch.stack(tot)
        return {"mae": err.mean().item(), "rel_mae": (err.mean() / tot.mean()).item(),
                "mae_k0": err[:, ~k_eval].mean().item(), "mae_k1": err[:, k_eval].mean().item()}

    first_set = next(iter(sets))
    if args.eval_only:
        net = ValueNet.from_checkpoint(torch.load(ckpt))
    else:
        torch.manual_seed(v["seed"])
        net = ValueNet(env.nx, reward.psi_dim, v["hidden"], v["out_dim"])
        stage = lambda x, u, p: reward.stage_cost(env, x, u, p)  # noqa: E731
        exclude = v["random_goals"]["holdout_box"] if mode == "random" else None
        sample_psi = make_psi_sampler(cfg, env, reward, mode, exclude=exclude)

        def goal_states(psi, gen):
            g = KeyDoor2D.goal_from_psi(reward, psi)
            k = (torch.rand(len(psi), 1, generator=gen) < 0.5).float()
            return torch.cat([g, k], -1)

        fvi = FittedValueIteration(env, stage, net, fvi_config(cfg, v, args.iterations),
                                   sample_states=env.sample_states, sample_psi=sample_psi, goal_states=goal_states,
                                   sample_starts=lambda n, g: env.sample_start_states(n, g))
        history = fvi.run(callback=lambda log, n: errors(n, first_set, max_goals=4))
        torch.save({"state_dict": net.state_dict(), "net_config": net.config(), "goal_mode": mode,
                    "config": cfg}, ckpt)
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

    # ------------------------------------------------------------------ #
    # value error per test set
    # ------------------------------------------------------------------ #
    err_rows = []
    for name in sets:
        e = errors(net, name)
        mean_exact = torch.stack([V(x_eval) for V in exact[name]]).mean().item()
        err_rows.append({"test_set": name, "goals": len(sets[name]), "mean_exact_value": mean_exact, **e})
        print(f"value error [{name}]: " + "  ".join(f"{k} {x:.4f}" for k, x in e.items())
              + f"  (mean exact value {mean_exact:.3f})")
    with open(out / "value_error.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(err_rows[0]))
        w.writeheader()
        w.writerows(err_rows)

    # heat maps
    psi0, V0 = sets[first_set][0], exact[first_set][0]
    goal0 = KeyDoor2D.goal_from_psi(reward, psi0).tolist()
    vmax = float(V0.V.max())
    with torch.no_grad():
        plot_value_heatmaps(str(out / "value_learned.png"), env, lambda x: net(x, psi0), goal0,
                            "Learned value V_theta", vmax=vmax)
        plot_value_heatmaps(str(out / "value_error.png"), env, lambda x: (net(x, psi0) - V0(x)).abs(), goal0,
                            "|V_theta - V_exact|", cmap="magma", label="absolute error")
    entries = []
    for name in sets:
        for i in range(min(2, len(sets[name]))):
            psi, V = sets[name][i], exact[name][i]
            gw = reward.unpack(psi)["goal"][0].item()
            g = KeyDoor2D.goal_from_psi(reward, psi).tolist()
            entries.append((f"{name}: goal ({g[0]:.2f}, {g[1]:.2f}), weight {gw:.2f}", g, V,
                            (lambda p: (lambda x: net(x, p)))(psi)))
    plot_value_comparison(str(out / "value_comparison.png"), env, entries)

    # ------------------------------------------------------------------ #
    # closed loop: short-horizon MPC with no / learned / exact terminal value
    # ------------------------------------------------------------------ #
    n = ev["num_episodes"]
    x0 = env.sample_start_states(n, seeded(ev["seed"]))
    learned = lambda x, p: net(x, p)  # noqa: E731
    rows = []
    Hs, names = v["eval_horizons"], ["none", "learned", "exact"]
    for set_name, psis in sets.items():
        idx = torch.arange(n) % len(psis)
        psi = psis[idx]
        oracle = OracleTerminal([exact[set_name][i] for i in idx.tolist()])
        oracle_one = OracleTerminal([exact[set_name][0]])
        goals = KeyDoor2D.goal_from_psi(reward, psi)
        paths = {}
        for H in Hs:
            for name in names:
                term = {"none": None, "learned": learned, "exact": oracle}[name]
                term1 = {"none": None, "learned": learned, "exact": oracle_one}[name]
                spec = {"horizon": H}
                st = run_episodes(env, reward, build_controller(env, reward, cfg, spec, term), x0, psi,
                                  ev["max_steps"])
                s = st.summary()
                s["ms_per_step"] = 1e3 * time_per_step(build_controller(env, reward, cfg, spec, term1), env, x0, psi,
                                                        ev["timing_steps"])
                rows.append({"test_set": set_name, "horizon": H, "terminal": name, **s})
                paths[(H, name)] = st
                print(f"[{set_name}] H={H:3d} terminal={name:8s} success {s['success_rate']:.2f}  "
                      f"steps {s['mean_steps_success']:.1f}  cost {s['mean_cost']:.2f}  {s['ms_per_step']:.2f} ms/step")

        fig, axes = plt.subplots(len(names), len(Hs), figsize=(4.2 * len(Hs), 4.3 * len(names)), squeeze=False)
        single_goal = len(psis) == 1
        for i, name in enumerate(names):
            for j, H in enumerate(Hs):
                st, ax = paths[(H, name)], axes[i, j]
                draw_keydoor(ax, env, goals[0].tolist() if single_goal else None)
                plot_trajectories(ax, st.traj[:12], st.success[:12], None if single_goal else goals[:12])
                ax.set_title(f"H={H}, terminal: {name}\n{st.success.float().mean():.0%} success (all {n})",
                             fontsize=10)
        fig.savefig(out / f"paths_{set_name}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    with open(out / "closed_loop.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
