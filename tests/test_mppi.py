import torch

from rloc.core.mppi import MPPI, MPCController, MPPIConfig, linear_interp_matrix


def point_step(x, u):
    return x + 0.1 * u.clamp(-1, 1)


def quad_cost(x, u, psi):  # psi = target
    return ((x - psi) ** 2).sum(-1) + 0.01 * (u ** 2).sum(-1)


def make(cfg=None, terminal=None):
    return MPPI(point_step, quad_cost, -torch.ones(2), torch.ones(2), cfg or MPPIConfig(horizon=10),
                terminal_cost_fn=terminal, generator=torch.Generator().manual_seed(0))


def test_interp_matrix_identity_and_hold():
    k = torch.arange(5, dtype=torch.float32)
    assert torch.allclose(linear_interp_matrix(k, k), torch.eye(5))
    M = linear_interp_matrix(torch.tensor([0.5, 10.0]), k)
    assert torch.allclose(M @ torch.arange(5.0), torch.tensor([0.5, 4.0]))


def test_shift_per_step_moves_sequence_forward():
    m = make()
    theta = torch.arange(10.0).reshape(1, 10, 1).expand(1, 10, 2)
    shifted = m.shift(theta)
    assert torch.allclose(shifted[0, :, 0], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 9.0]))


def test_rollout_cost_matches_manual():
    m = make()
    x0, psi = torch.zeros(1, 2), torch.tensor([[1.0, 0.0]])
    U = torch.ones(1, 1, 10, 2) * torch.tensor([1.0, 0.0])
    X, c = m.rollout(x0, U, psi)
    xs = torch.stack([torch.tensor([0.1 * (t + 1), 0.0]) for t in range(10)])
    assert torch.allclose(X[0, 0, 1:], xs)
    assert torch.allclose(c[0, 0], quad_cost(xs, U[0, 0], psi[0]).sum())


def test_closed_loop_reaches_targets_batched():
    m = make(MPPIConfig(horizon=10, num_samples=128))
    ctrl = MPCController(m)
    targets = torch.tensor([[0.8, 0.3], [-0.5, -0.7], [0.0, 0.9]])
    x = torch.zeros(3, 2)
    for _ in range(40):
        x = point_step(x, ctrl.act(x, targets))
    assert ((x - targets).norm(dim=-1) < 0.05).all()


def test_knots_and_best_cost():
    m = make(MPPIConfig(horizon=20, num_knots=5, num_samples=256, num_iters=5))
    res = m.optimize(torch.zeros(2, 2), torch.tensor([[1.0, 1.0], [-1.0, 0.5]]))
    assert res.controls.shape == (2, 20, 2)
    assert (res.best_cost <= res.cost + 1e-6).all()
    # first controls should point towards the targets
    assert (res.controls[0, 0] > 0).all() and res.controls[1, 0, 0] < 0


def test_terminal_cost_is_used():
    # terminal cost pulls to (+1, 0) while the stage cost is zero
    m = MPPI(point_step, lambda x, u, p: torch.zeros(x.shape[:-1]), -torch.ones(2), torch.ones(2),
             MPPIConfig(horizon=5, num_samples=256, num_iters=5),
             terminal_cost_fn=lambda x, p: ((x - torch.tensor([1.0, 0.0])) ** 2).sum(-1),
             generator=torch.Generator().manual_seed(1))
    res = m.optimize(torch.zeros(1, 2), torch.zeros(1, 1))
    assert res.controls[0, :, 0].mean() > 0.5
