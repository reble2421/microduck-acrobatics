"""Headspin (Mjlab-HeadSpin-Flat-MicroDuck): config invariants, the spin command sampler,
and the headspin rewards on a minimal fake env. CPU only, no warp env is built.
"""

import math
from types import SimpleNamespace

import numpy as np
import torch

from mjlab_microduck.tasks import headspin_mdp as hsp
from mjlab_microduck.tasks import headstand_mdp as hs
from mjlab_microduck.tasks.microduck_headspin_env_cfg import (
    COMMAND_STAGES,
    MicroduckHeadSpinRlCfg,
    make_microduck_headspin_env_cfg,
)

TASK = "Mjlab-HeadSpin-Flat-MicroDuck"
SPIN_TERMS = ("spin_when_stopped", "still_when_stopped", "headspin_track")


def test_task_is_registered():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks, load_rl_cfg

    assert TASK in list_tasks()
    rl = load_rl_cfg(TASK)
    assert rl.experiment_name == "microduck_headstand" and rl.run_name == "microduck_headspin"
    assert MicroduckHeadSpinRlCfg.algorithm.symmetry_cfg is None


def test_reward_set_and_signs():
    cfg = make_microduck_headspin_env_cfg()
    assert list(cfg.rewards)[-3:] == list(SPIN_TERMS)          # appended in this order
    for name in ("headstand_hold", "headspin_track", "still_when_stopped"):
        assert cfg.rewards[name].weight > 0, name
    assert cfg.rewards["spin_when_stopped"].weight < 0
    assert "headspin_track" in make_microduck_headspin_env_cfg(play=True).rewards
    assert list(cfg.events)[-2:] == ["reset_com_velocity", "reset_support_velocity"]


def test_command_settings():
    cfg = make_microduck_headspin_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, hsp.HeadSpinCommandCfg)
    assert cmd.ranges.lin_vel_x == (0.0, 0.0) and cmd.ranges.lin_vel_y == (0.0, 0.0)
    assert cmd.ranges.ang_vel_z[1] == COMMAND_STAGES[0]["ang_vel_range"]
    assert cmd.rel_turn_in_place_envs == 0.0
    assert cmd.zero_frac == 0.25 and cmd.min_frac == 0.5
    angs = [s["ang_vel_range"] for s in COMMAND_STAGES]
    assert angs == sorted(angs) and angs[-1] <= 3.0
    assert cfg.curriculum["headstand_command_ranges"].params["velocity_stages"] == COMMAND_STAGES
    assert cfg.curriculum["action_rate_weight"].params["weight_stages"] == [{"step": 0, "weight": -0.1}]
    assert cfg.curriculum["neck_torque_weight"].params["weight_stages"] == [{"step": 0, "weight": 0.0}]


def test_spin_rates_are_never_tiny_and_both_ways():
    torch.manual_seed(0)
    w = hsp.sample_spin_rates(4000, 3.0, 0.5, "cpu")
    assert float(w.abs().min()) >= 1.5 - 1e-6 and float(w.abs().max()) <= 3.0 + 1e-6
    assert 0.45 < float((w > 0).float().mean()) < 0.55


# ---- fake env -------------------------------------------------------------------------

def _quat_from_to(u, v):
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    axis = np.cross(u, v)
    s, c = np.linalg.norm(axis), float(u @ v)
    if s < 1e-9:
        return np.array([1.0, 0, 0, 0])
    ang = math.atan2(s, c)
    return np.array([math.cos(ang / 2), *(math.sin(ang / 2) * axis / s)])


HEAD, AL, AR = 0, 1, 2
ROOT = 1
TOP_DOWN_Q = _quat_from_to(hs.HEAD_TOP_AXIS, [0, 0, -1])


class _Sensor:
    def __init__(self, n):
        self.data = SimpleNamespace(found=torch.zeros(n, 1))


class _Scene(dict):
    pass


class _Asset:
    NAMES = {"jaw_soft": HEAD, "ankle_left": AL, "ankle_right": AR}

    def __init__(self, n):
        self.indexing = SimpleNamespace(root_body_id=ROOT)
        self.data = SimpleNamespace(
            body_link_quat_w=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 3, 1),
            body_link_lin_vel_w=torch.zeros(n, 3, 3),
            body_link_ang_vel_w=torch.zeros(n, 3, 3),
            body_link_pos_w=torch.zeros(n, 3, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1),
        )

    def find_bodies(self, name):
        return [self.NAMES[name]], [name]


def _run(command, wz=0.0, com_v=(0.0, 0.0), steps=200, on_head=True):
    n = 1
    asset = _Asset(n)
    scene = _Scene(robot=asset)
    scene.sensors = {hs.HEAD_SENSOR: _Sensor(n), hs.NONHEAD_SENSOR: _Sensor(n)}
    scene.terrain = SimpleNamespace(env_origins=torch.zeros(n, 3))
    cmd = torch.tensor([command], dtype=torch.float32)
    env = SimpleNamespace(num_envs=n, device="cpu", step_dt=0.02, common_step_counter=0,
                          scene=scene, command_manager=SimpleNamespace(get_command=lambda name: cmd),
                          sim=SimpleNamespace(data=SimpleNamespace(subtree_com=torch.zeros(n, 3, 3))))
    asset.data.body_link_quat_w[:, HEAD] = torch.tensor(TOP_DOWN_Q, dtype=torch.float32)
    asset.data.projected_gravity_b[:] = torch.tensor(hs.HEADSTAND_GRAVITY_B)
    asset.data.body_link_pos_w[:, AL, 2] = 0.10
    asset.data.body_link_pos_w[:, AR, 2] = 0.10
    scene.sensors[hs.HEAD_SENSOR].data.found[:] = 1.0
    if not on_head:
        scene.sensors[hs.NONHEAD_SENSOR].data.found[:] = 1.0
    asset.data.body_link_ang_vel_w[:, HEAD, 2] = wz
    vc = torch.tensor(com_v)
    for _ in range(steps):
        env.common_step_counter += 1
        env.sim.data.subtree_com[:, ROOT, :2] += vc * env.step_dt
        out = {"spin_cost": float(hsp.spin_when_stopped(env)[0]),       # reward order of the task
               "still": float(hsp.still_when_stopped(env)[0]),
               "track": float(hsp.headspin_track(env)[0])}
    return out


def _close(a, b, tol=0.02):
    return abs(a - b) <= tol


def test_spin_tracking():
    assert _close(_run([0.0, 0.0, 2.0], wz=2.0)["track"], 1.0)
    assert _close(_run([0.0, 0.0, -2.0], wz=-1.0)["track"], 0.5)
    assert _close(_run([0.0, 0.0, 2.0], wz=-2.0)["track"], -0.25)   # wrong way: floor
    assert _close(_run([0.0, 0.0, 2.0], wz=4.0)["track"], 0.0)      # overspeed: no pay
    assert _close(_run([0.0, 0.0, 2.0])["track"], 0.0)              # not spinning: nothing


def test_wandering_while_spinning_does_not_pay():
    assert _close(_run([0.0, 0.0, 2.0], wz=2.0, com_v=(0.03, 0.0))["track"], 0.0)
    assert _close(_run([0.0, 0.0, 2.0], wz=2.0, com_v=(0.015, 0.0))["track"], 0.5)


def test_stop_on_command():
    r = _run([0.0, 0.0, 0.0], wz=2.0)
    assert r["track"] == 0.0 and _close(r["spin_cost"], 1.0) and r["still"] < 1e-6
    assert _run([0.0, 0.0, 2.0], wz=2.0)["spin_cost"] == 0.0         # no cost while told to spin
    still = _run([0.0, 0.0, 0.0])                                     # told to stop, staying put
    assert _close(still["still"], 1.0) and still["spin_cost"] == 0.0


def test_nothing_is_paid_off_the_head():
    off = _run([0.0, 0.0, 2.0], wz=2.0, on_head=False)
    assert off["track"] == 0.0 and _run([0.0, 0.0, 0.0], on_head=False)["still"] == 0.0
