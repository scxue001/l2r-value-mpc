import torch

from rloc.envs.keydoor2d import KeyDoor2D
from rloc.envs.keydoor2d_dp import KeyDoorDP


def step(env, x, u):
    return env.step(torch.tensor([x]), torch.tensor([u]))[0]


def test_closed_door_blocks():
    env = KeyDoor2D()
    x = step(env, [-0.05, 0.0, 0.0], [1.0, 0.0])
    assert x[0] < env.cfg.wall_x  # stayed on the left
    # moving right and up at the wall keeps the y motion (slides)
    x = step(env, [-0.05, 0.0, 0.0], [1.0, 1.0])
    assert x[0] < env.cfg.wall_x and torch.isclose(x[1], torch.tensor(0.1))


def test_open_door_passes_only_through_gap():
    env = KeyDoor2D()
    x = step(env, [-0.05, 0.0, 1.0], [1.0, 0.0])
    assert x[0] > env.cfg.wall_x
    x = step(env, [-0.05, 0.5, 1.0], [1.0, 0.0])  # outside the gap: wall
    assert x[0] < env.cfg.wall_x
    x = step(env, [0.05, 0.0, 1.0], [-1.0, 0.0])  # also from the right
    assert x[0] < env.cfg.wall_x


def test_switch_sets_key_and_stays():
    env = KeyDoor2D()
    sx, sy = env.cfg.switch_pos
    x = step(env, [sx - 0.1, sy, 0.0], [1.0, 0.0])
    assert x[2] == 1.0
    x = step(env, [0.9, -0.9, 1.0], [0.0, 0.0])  # far away, key kept
    assert x[2] == 1.0


def test_room_bounds_and_control_clip():
    env = KeyDoor2D()
    x = step(env, [0.95, 0.95, 0.0], [5.0, 5.0])
    assert torch.allclose(x[:2], torch.tensor([1.0, 1.0]))


def test_batched_shapes():
    env = KeyDoor2D()
    x = env.sample_states(12).reshape(3, 4, 3)
    u = torch.zeros(3, 4, 2)
    assert env.step(x, u).shape == (3, 4, 3)
    starts = env.sample_start_states(50, torch.Generator().manual_seed(0))
    assert (starts[:, 0] < env.cfg.wall_x).all() and (starts[:, 2] == 0).all()


def test_goal_sampling_region_and_exclusion():
    env = KeyDoor2D()
    box = {"x": [0.55, 0.75], "y": [0.2, 0.5]}
    g = env.sample_goals(500, torch.Generator().manual_seed(0), exclude=box)
    inside = (g[:, 0] >= 0.55) & (g[:, 0] <= 0.75) & (g[:, 1] >= 0.2) & (g[:, 1] <= 0.5)
    assert len(g) == 500 and not inside.any()
    h = env.sample_goals(50, torch.Generator().manual_seed(1), region=box)
    assert (h[:, 0] >= 0.55).all() and (h[:, 0] <= 0.75).all() and (h[:, 1] >= 0.2).all() and (h[:, 1] <= 0.5).all()


def test_make_psi_batched_weights():
    env = KeyDoor2D()
    r = env.default_reward()
    goals = torch.tensor([[0.5, 0.0], [0.8, 0.3]])
    psi = env.make_psi(r, goals, torch.tensor([0.5, 2.0]), 0.01)
    assert psi.shape == (2, r.psi_dim)
    assert torch.allclose(env.goal_from_psi(r, psi), goals)
    assert torch.allclose(r.unpack(psi)["goal"][0], torch.tensor([0.5, 2.0]))


def test_switch_shaping_static_and_staged():
    env = KeyDoor2D()
    sx, sy = env.cfg.switch_pos
    x = torch.tensor([[sx + 0.3, sy, 0.0], [sx + 0.3, sy, 1.0], [sx, sy, 0.0]])
    static = env.switch_shaping(x, weight=2.0, staged=False)
    staged = env.switch_shaping(x, weight=2.0, staged=True)
    expected = 2.0 * 0.09 * env.dt
    assert torch.allclose(static[:2], torch.tensor([expected, expected]))
    assert torch.isclose(staged[0], torch.tensor(expected)) and staged[1] == 0.0  # off once pressed
    assert static[2] == 0.0 and staged[2] == 0.0  # zero on the switch


def test_extra_cost_reaches_planner_only():
    from rloc.eval.common import build_controller, load_config, ROOT

    cfg = load_config(ROOT / "configs/keydoor.yaml")
    env = KeyDoor2D()
    r = env.default_reward()
    calls = []

    def extra(x, u, p):
        calls.append(x.shape)
        return torch.zeros(x.shape[:-1])

    ctrl = build_controller(env, r, cfg, {"horizon": 3}, extra_cost=extra)
    ctrl.act(torch.tensor([[-0.5, 0.0, 0.0]]), env.make_psi(r, [0.8, 0.0])[None])
    assert calls, "extra cost was not used by the planner"


def test_exact_value_sanity():
    env = KeyDoor2D()
    r = env.default_reward()
    psi = env.make_psi(r, [0.8, 0.0])
    dp = KeyDoorDP(env, r, psi)
    V = dp.solve()
    assert (V.V >= 0).all()
    pts = torch.tensor([[0.8, 0.0, 0.0], [-0.5, -0.5, 0.0], [-0.5, -0.5, 1.0]])
    v = V(pts)
    assert v[0] < 1e-3  # at the goal
    assert v[1] > v[2]  # a closed door can only make things worse
    # k = 1 never worse than k = 0 anywhere (up to interpolation noise)
    assert (V.V[1] <= V.V[0] + 1e-4).all()
