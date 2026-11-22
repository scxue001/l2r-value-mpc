import torch

from rloc.core.fvi import FittedValueIteration, FVIConfig
from rloc.core.mppi import MPPIConfig
from rloc.core.value_net import ValueNet


class Line:
    """1D point x' = x + 0.1 u, |u| <= 1, goal at psi."""

    nx, nu, dt = 1, 1, 0.1
    u_low, u_high = -torch.ones(1), torch.ones(1)

    def step(self, x, u):
        return x + self.dt * u.clamp(-1, 1)


def test_value_net_shapes_nonneg_and_checkpoint():
    net = ValueNet(3, 4, hidden=(16, 16), out_dim=8)
    x, psi = torch.randn(5, 7, 3), torch.randn(4)
    v = net(x, psi)
    assert v.shape == (5, 7) and (v >= 0).all()
    net.set_normalization(torch.randn(100, 3), torch.ones(100, 4))  # constant psi -> std stays 1
    assert torch.allclose(net.in_std[3:], torch.ones(4))
    net2 = ValueNet.from_checkpoint({"state_dict": net.state_dict(), "net_config": net.config()})
    assert torch.allclose(net2(x, psi), net(x, psi))


def test_fvi_rollout_states_mix():
    env = Line()
    stage = lambda x, u, p: ((x - p) ** 2).sum(-1) * env.dt  # noqa: E731
    cfg = FVIConfig(iterations=1, states_per_iter=100, rollout_frac=0.3, rollout_starts=8, rollout_steps=5,
                    rollout_horizon=3, train_steps=5,
                    target=MPPIConfig(horizon=3, num_samples=16, num_iters=1, update="best"))
    fvi = FittedValueIteration(env, stage, ValueNet(1, 1, (8,), 4), cfg,
                               sample_states=lambda n, g: 2 * torch.rand(n, 1, generator=g) - 1,
                               sample_psi=lambda n, g: torch.zeros(n, 1),
                               sample_starts=lambda n, g: torch.ones(n, 1))
    x, psi = fvi.rollout_states(30)
    assert x.shape == (30, 1) and psi.shape == (30, 1)
    assert (x < 1.0).all()  # the controller moved every start (x = 1) towards the goal
    fvi.step()
    assert fvi.history[0].iteration == 1


def test_fvi_learns_line_value():
    env = Line()
    stage = lambda x, u, p: ((x - p) ** 2).sum(-1) * env.dt  # noqa: E731
    net = ValueNet(1, 1, hidden=(32, 32), out_dim=8)
    cfg = FVIConfig(iterations=8, states_per_iter=512,
                    target=MPPIConfig(horizon=5, num_samples=64, num_iters=2, noise_sigma=1.0, update="best"),
                    train_steps=300, batch_size=256, lr=3e-3, anchor_states=64)
    fvi = FittedValueIteration(
        env, stage, net, cfg,
        sample_states=lambda n, g: 2 * torch.rand(n, 1, generator=g) - 1,
        sample_psi=lambda n, g: torch.zeros(n, 1),
        goal_states=lambda psi, g: psi.clone(),
    )
    fvi.run(verbose=False)
    xs = torch.tensor([[0.0], [0.5], [1.0]])
    v = net(xs, torch.zeros(1))
    assert v[0] < 0.02
    assert v[0] < v[1] < v[2]
    # exact value with |u| = 1 to the goal: sum of dt * x_t^2 ~ |x|^3 / 3 -> about 0.33 - 0.38 at x = 1
    assert 0.2 < v[2] < 0.5
