"""Backward roll (Mjlab-BackRoll-Flat-MicroDuck): the roulade with the roll
direction flipped and its two progress windows mirrored. Everything else must
stay identical to the roulade, and the roulade itself must be unchanged.

CPU only; the reward/reset checks drive the real mdp functions with a minimal
fake env.
"""

import math
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_backroll_env_cfg import (
    BACKROLL_HEAD_LATCH,
    BACKROLL_HEAD_PIVOT,
    MicroduckBackrollRlCfg,
    make_microduck_backroll_env_cfg,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)

TWO_PI = 2 * math.pi


# ---- configuration ----------------------------------------------------------

def test_backroll_rolls_backward_and_roulade_keeps_the_default():
    for play in (False, True):
        back = make_microduck_backroll_env_cfg(play=play).events["set_roulade_state"].params
        fwd = make_microduck_roulade_env_cfg(play=play).events["set_roulade_state"].params
        assert back["direction"] == -1.0
        assert "direction" not in fwd  # forward default (+1) untouched


def test_windows_are_the_mirror_of_the_roulade():
    # backward progress p passes the same orientations as forward 360 - p
    lo, hi = BACKROLL_HEAD_LATCH
    assert math.isclose(lo, TWO_PI - microduck_mdp._HEAD_LATCH_HI)
    assert math.isclose(hi, TWO_PI - microduck_mdp._HEAD_LATCH_LO)
    fwd = make_microduck_roulade_env_cfg().rewards["roulade_head_pivot"].params
    b_lo, b_hi = BACKROLL_HEAD_PIVOT
    assert math.isclose(b_lo, TWO_PI - fwd["angle_hi"])
    assert math.isclose(b_hi, TWO_PI - fwd["angle_lo"])
    back = make_microduck_backroll_env_cfg()
    assert back.rewards["roulade_head_pivot"].params["angle_lo"] == b_lo
    assert back.events["set_roulade_state"].params["head_latch_lo"] == lo


def test_everything_else_matches_the_roulade():
    b, f = make_microduck_backroll_env_cfg(), make_microduck_roulade_env_cfg()
    assert set(b.rewards) == set(f.rewards)
    for name, term in f.rewards.items():
        assert b.rewards[name].weight == term.weight, name
    assert set(b.events) == set(f.events)
    assert set(b.curriculum) == set(f.curriculum)
    assert b.episode_length_s == f.episode_length_s


def test_runner_logs_to_its_own_experiment():
    assert MicroduckBackrollRlCfg.experiment_name == "microduck_backroll"
    assert MicroduckRouladeRlCfg.experiment_name == "microduck_roulade"
    assert MicroduckBackrollRlCfg.max_iterations == MicroduckRouladeRlCfg.max_iterations


def test_task_is_registered():
    import mjlab_microduck.tasks  # noqa: F401  (runs register_mjlab_task)
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    assert "Mjlab-BackRoll-Flat-MicroDuck" in list_tasks()
    env_cfg = load_env_cfg("Mjlab-BackRoll-Flat-MicroDuck")
    assert env_cfg.events["set_roulade_state"].params["direction"] == -1.0
    assert load_rl_cfg("Mjlab-BackRoll-Flat-MicroDuck").experiment_name == "microduck_backroll"


# ---- rotation accumulator (reward side) --------------------------------------

class _Sensor:
    def __init__(self, n):
        self.data = SimpleNamespace(found=torch.ones(n, 1))


class _Asset:
    def __init__(self, n):
        self.data = SimpleNamespace(
            root_link_ang_vel_b=torch.zeros(n, 3),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
            # head turned 180 deg about y: its flat top faces the floor
            body_link_quat_w=torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(n, 1).unsqueeze(1),
        )

    def find_bodies(self, name):
        return [0], [name]


def _accum_env(n=1):
    sensors = {"robot_ground_contact": _Sensor(n), "head_ground_contact": _Sensor(n)}
    env = SimpleNamespace(num_envs=n, device="cpu", step_dt=0.02, common_step_counter=0,
                          scene=SimpleNamespace(sensors=sensors))
    return env, _Asset(n)


def _spin(env, asset, omega_y, steps):
    asset.data.root_link_ang_vel_b[:, 1] = omega_y
    for _ in range(steps):
        env.common_step_counter += 1
        microduck_mdp._update_roulade_accum(env, asset)


def test_forward_roll_accumulates_positive_pitch_rate_by_default():
    env, asset = _accum_env()
    _spin(env, asset, +2.0, 10)
    assert math.isclose(float(env._roulade_accum[0]), 0.4, rel_tol=1e-4)


def test_backward_roll_accumulates_negative_pitch_rate():
    env, asset = _accum_env()
    env._roulade_direction = -1.0
    _spin(env, asset, -2.0, 10)
    assert math.isclose(float(env._roulade_accum[0]), 0.4, rel_tol=1e-4)
    _spin(env, asset, +2.0, 5)  # rocking the other way un-counts ...
    assert float(env._roulade_accum[0]) < 0.4
    assert math.isclose(float(env._roulade_max[0]), 0.4, rel_tol=1e-4)  # ... frontier stays


def _latch_at(progress_deg, window):
    env, asset = _accum_env()
    microduck_mdp._roulade_state(env)
    env._roulade_accum[:] = math.radians(progress_deg)
    env._roulade_max[:] = math.radians(progress_deg)
    if window is not None:
        env._roulade_latch_window = window
    _spin(env, asset, 0.0, 1)
    return bool(env._roulade_head_latch[0])


def test_head_latch_window_follows_the_task():
    # head top on the floor at 250 deg of progress = a backward roll's head phase
    assert _latch_at(250, BACKROLL_HEAD_LATCH) is True
    assert _latch_at(250, None) is False  # roulade window is 20..170 deg
    assert _latch_at(90, None) is True  # roulade still latches early
    assert _latch_at(90, BACKROLL_HEAD_LATCH) is False


# ---- reset / mid-roll spawns ---------------------------------------------------

class _JointAsset:
    def find_joints(self, pattern):
        return list(range(14)), [f"j{i}" for i in range(14)]


class _Scene(dict):
    sensors = {}


def _spawn(direction, n=512, **extra):
    torch.manual_seed(0)
    data = SimpleNamespace(qpos=torch.zeros(n, 7 + 14), qvel=torch.zeros(n, 6 + 14))
    env = SimpleNamespace(num_envs=n, device="cpu", sim=SimpleNamespace(data=data),
                          scene=_Scene(robot=_JointAsset()))
    microduck_mdp.reset_roulade_state(
        env, torch.arange(n), standing_prob=0.0, midroll_prob=1.0,
        midroll_pitch_min=math.radians(50.0), midroll_pitch_max=math.radians(340.0),
        midroll_omega_range=(1.0, 3.0), direction=direction, **extra)
    return env


def _body_z(quat):
    w, x, y, z = quat.unbind(-1)
    return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)


def test_backward_spawns_are_pitched_and_spun_the_other_way():
    fwd = _spawn(+1.0)
    back = _spawn(-1.0, head_latch_lo=BACKROLL_HEAD_LATCH[0], head_latch_hi=BACKROLL_HEAD_LATCH[1])
    zf, zb = _body_z(fwd.sim.data.qpos[:, 3:7]), _body_z(back.sim.data.qpos[:, 3:7])
    # same random draws, pitch mirrored: vertical part equal, fore/aft lean reversed
    assert torch.allclose(zf[:, 2], zb[:, 2], atol=1e-5)
    clear = torch.sin(back._roulade_accum).abs() > 0.3
    lean = (zf[:, :2] * zb[:, :2]).sum(-1)
    assert bool((lean[clear] < 0).all())
    assert bool((fwd.sim.data.qvel[:, 4] >= 1.0).all())
    assert bool((back.sim.data.qvel[:, 4] <= -1.0).all())
    # progress bookkeeping is direction-free: both start at the spawn angle
    assert torch.allclose(fwd._roulade_accum, back._roulade_accum)


def test_mid_roll_spawns_are_prelatched_only_past_the_window_start():
    fwd = _spawn(+1.0)
    assert bool(fwd._roulade_head_latch.all())  # roulade: unchanged, all pre-latched
    back = _spawn(-1.0, head_latch_lo=BACKROLL_HEAD_LATCH[0], head_latch_hi=BACKROLL_HEAD_LATCH[1])
    assert torch.equal(back._roulade_head_latch, back._roulade_accum >= BACKROLL_HEAD_LATCH[0])
