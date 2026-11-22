import pytest
import torch

from rloc.core.reward_api import ControlEffort, Distance, PositionTarget, Reward
from rloc.envs.keydoor2d import KeyDoor2D


def test_pack_unpack_roundtrip():
    r = Reward({"goal": PositionTarget("point", 2), "near": Distance("point", "switch"), "effort": ControlEffort()})
    assert r.psi_dim == (1 + 2) + 1 + 1
    psi = r.pack({"goal": {"weight": 2.0, "target": [0.3, -0.4]}, "effort": {"weight": 0.1}})
    parts = r.unpack(psi)
    assert parts["goal"][0].item() == 2.0
    assert torch.allclose(parts["goal"][1]["target"], torch.tensor([0.3, -0.4]))
    assert parts["near"][0].item() == 0.0  # missing term -> disabled
    assert parts["effort"][0].item() == pytest.approx(0.1)


def test_pack_batch_matches_pack():
    r = Reward({"goal": PositionTarget("point", 2), "near": Distance("point", "switch"), "effort": ControlEffort()})
    goals = torch.tensor([[0.3, -0.4], [0.7, 0.1], [0.5, 0.5]])
    weights = torch.tensor([0.5, 1.0, 2.0])
    batch = r.pack_batch(3, {"goal": {"weight": weights, "target": goals}, "effort": {"weight": 0.01}})
    for i in range(3):
        single = r.pack({"goal": {"weight": weights[i].item(), "target": goals[i]}, "effort": {"weight": 0.01}})
        assert torch.allclose(batch[i], single)
    # shared target for all rows
    shared = r.pack_batch(2, {"goal": {"weight": 1.0, "target": [0.1, 0.2]}})
    assert torch.allclose(shared[1, 1:3], torch.tensor([0.1, 0.2]))


def test_unknown_term_rejected():
    r = Reward({"goal": PositionTarget("point", 2)})
    with pytest.raises(KeyError):
        r.pack({"nope": {"weight": 1.0}})


def test_stage_cost_value_and_broadcast():
    env = KeyDoor2D()
    r = env.default_reward()
    psi = env.make_psi(r, [0.5, 0.0], goal_weight=2.0, effort_weight=0.5)
    x = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 1.0]])
    u = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    c = r.stage_cost(env, x, u, psi)
    expected = torch.tensor([2.0 * 0.25 + 0.5 * 1.0, 0.0]) * env.dt
    assert torch.allclose(c, expected)
    # batched psi with extra leading dims
    c2 = r.stage_cost(env, x[None].expand(3, 2, 3), u[None].expand(3, 2, 2), psi.expand(3, 2, -1))
    assert c2.shape == (3, 2) and torch.allclose(c2[1], expected)
