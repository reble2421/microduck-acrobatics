"""Headstand task (Mjlab-HeadStand-Flat-MicroDuck): config invariants, the
headstand gate / shaping terms and the averaged head velocity on a minimal fake
env, the spawn buckets, and the balance pose itself on the plain MuJoCo model.

CPU only, no warp env is built.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from mjlab_microduck.tasks import headstand_mdp as hs
from mjlab_microduck.tasks.microduck_headstand_env_cfg import (
    COMMAND_STAGES,
    ROULADE_ONLY_REWARDS,
    MicroduckHeadstandRlCfg,
    make_microduck_headstand_env_cfg,
)

TASK = "Mjlab-HeadStand-Flat-MicroDuck"
SCENE_XML = Path(__file__).resolve().parents[1] / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene.xml"


# ---- configuration ----------------------------------------------------------

def test_task_is_registered():
    import mjlab_microduck.tasks  # noqa: F401  (runs register_mjlab_task)
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    assert TASK in list_tasks()
    assert "set_headstand_state" in load_env_cfg(TASK).events
    assert load_rl_cfg(TASK).experiment_name == "microduck_headstand"


def test_reward_set_and_signs():
    cfg = make_microduck_headstand_env_cfg()
    for name in ROULADE_ONLY_REWARDS:
        assert name not in cfg.rewards, name
    for name in ("headstand_hold", "headstand_entry_orient", "headstand_entry_lift"):
        assert cfg.rewards[name].weight > 0, name
    for name in ("action_rate_l2", "dof_pos_limits", "self_collisions", "body_ang_vel"):
        assert cfg.rewards[name].weight < 0, name
    # self-negating |a_z| penalty: POSITIVE weight is the penalty
    assert cfg.rewards["gentle_landing"].weight > 0
    stages = cfg.curriculum["neck_torque_weight"].params["weight_stages"]
    assert all(s["weight"] <= 0 for s in stages)


def test_curricula_point_at_live_terms():
    cfg = make_microduck_headstand_env_cfg()
    for name, term in cfg.curriculum.items():
        p = term.params
        if "reward_name" in p:
            assert p["reward_name"] in cfg.rewards, name
        if "event_name" in p:
            assert p["event_name"] in cfg.events, name
    assert "set_roulade_state" not in cfg.events
    mix = cfg.curriculum["headstand_spawn_mix"].params["param_stages"]
    assert mix[-1]["params"]["standing_prob"] > mix[0]["params"]["standing_prob"]


def test_command_settings():
    cmd = make_microduck_headstand_env_cfg().commands["twist"]
    assert cmd.rel_forward_envs == 0.0  # the base default would force vx >= 0.3
    assert cmd.heading_command is False
    assert 0.0 < cmd.rel_standing_envs < 1.0
    # the command stays in the observation but asks for nothing: hold still
    assert all(s["lin_vel_range"] == 0.0 and s["ang_vel_range"] == 0.0 for s in COMMAND_STAGES)
    assert cmd.ranges.lin_vel_x[1] == 0.0 and cmd.ranges.ang_vel_z[1] == 0.0


def test_runner():
    assert MicroduckHeadstandRlCfg.experiment_name == "microduck_headstand"
    assert MicroduckHeadstandRlCfg.algorithm.symmetry_cfg is None
    assert make_microduck_headstand_env_cfg().episode_length_s == 12.0


# ---- geometry helpers -----------------------------------------------------------

def _quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def _quat_from_to(u, v):
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    axis = np.cross(u, v)
    s, c = np.linalg.norm(axis), float(u @ v)
    if s < 1e-9:
        return np.array([1.0, 0, 0, 0])
    ang = math.atan2(s, c)
    return np.array([math.cos(ang / 2), *(math.sin(ang / 2) * axis / s)])


def test_headstand_constants_are_consistent():
    q = np.array(hs.HEADSTAND_QUAT)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-4
    g_b = _quat_to_mat(q).T @ np.array([0.0, 0.0, -1.0])
    assert np.allclose(g_b, hs.HEADSTAND_GRAVITY_B, atol=2e-3)


def test_heading_frame():
    yaw = torch.tensor([math.pi / 2])
    out = hs.to_heading_frame(torch.tensor([[0.0, 1.0]]), yaw)
    assert torch.allclose(out, torch.tensor([[1.0, 0.0]]), atol=1e-6)
    q = torch.tensor([[math.cos(0.3), 0.0, 0.0, math.sin(0.3)]])  # yaw 0.6 about z
    # head z-axis vertical here: rotate it onto +x first so the heading is defined
    q_x = torch.tensor(np.array([_quat_from_to([0, 0, 1], [1, 0, 0])]), dtype=torch.float32)
    assert abs(float(hs.head_heading(hs.quat_mul(q, q_x))[0]) - 0.6) < 1e-5


def test_potentials():
    stand = hs.orientation_potential(torch.tensor([[0.0, 0.0, -1.0]]), torch.tensor([1.0]))
    target = hs.orientation_potential(torch.tensor([hs.HEADSTAND_GRAVITY_B]), torch.tensor([-0.91]))
    faceplant = hs.orientation_potential(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([0.6]))
    assert float(stand[0]) == 0.0
    assert float(target[0]) > 0.99
    assert float(faceplant[0]) == 0.0  # neck at HOME: head top not down, no credit
    lift = hs.lift_potential(torch.tensor([0.023, 0.10, 0.10]), torch.tensor([True, True, False]))
    assert lift.tolist() == [0.0, 1.0, 0.0]


# ---- gate / shaping / averaged velocity on a fake env -------------------------------

HEAD, AL, AR = 0, 1, 2
TOP_DOWN_Q = _quat_from_to(hs.HEAD_TOP_AXIS, [0, 0, -1])


class _Sensor:
    def __init__(self, n):
        self.data = SimpleNamespace(found=torch.zeros(n, 1))


class _Scene(dict):
    pass


class _Asset:
    NAMES = {"jaw_soft": HEAD, "ankle_left": AL, "ankle_right": AR}

    def __init__(self, n):
        self.data = SimpleNamespace(
            body_link_quat_w=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 3, 1),
            body_link_lin_vel_w=torch.zeros(n, 3, 3),
            body_link_ang_vel_w=torch.zeros(n, 3, 3),
            body_link_pos_w=torch.zeros(n, 3, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1),
        )

    def find_bodies(self, name):
        return [self.NAMES[name]], [name]

    def find_joints(self, pattern):
        return list(range(14)), [f"j{i}" for i in range(14)]


def _fake_env(n=1):
    asset = _Asset(n)
    scene = _Scene(robot=asset)
    scene.sensors = {hs.HEAD_SENSOR: _Sensor(n), hs.NONHEAD_SENSOR: _Sensor(n)}
    scene.terrain = SimpleNamespace(env_origins=torch.zeros(n, 3))
    cmd = torch.zeros(n, 3)
    env = SimpleNamespace(num_envs=n, device="cpu", step_dt=0.02, common_step_counter=0,
                          scene=scene, command_manager=SimpleNamespace(get_command=lambda name: cmd))
    return env, asset, cmd


def _set_headstand(env, asset, feet_z=0.10):
    asset.data.body_link_quat_w[:, HEAD] = torch.tensor(TOP_DOWN_Q, dtype=torch.float32)
    asset.data.projected_gravity_b[:] = torch.tensor(hs.HEADSTAND_GRAVITY_B)
    asset.data.body_link_pos_w[:, AL, 2] = feet_z
    asset.data.body_link_pos_w[:, AR, 2] = feet_z
    env.scene.sensors[hs.HEAD_SENSOR].data.found[:] = 1.0
    env.scene.sensors[hs.NONHEAD_SENSOR].data.found[:] = 0.0


def _step(env):
    env.common_step_counter += 1
    return {
        "hold": float(hs.headstand_hold(env)[0]),
        "orient": float(hs.headstand_entry_orient(env)[0]),
        "lift": float(hs.headstand_entry_lift(env)[0]),
    }


def test_standing_earns_nothing():
    env, asset, _ = _fake_env()
    env.scene.sensors[hs.NONHEAD_SENSOR].data.found[:] = 1.0  # feet on the floor
    for _ in range(3):
        r = _step(env)
    assert r == {"hold": 0.0, "orient": 0.0, "lift": 0.0}


def test_entry_shaping_pays_once_and_is_rate_capped():
    env, asset, _ = _fake_env()
    env.scene.sensors[hs.NONHEAD_SENSOR].data.found[:] = 1.0
    _step(env)  # fresh: frontier initialised at the standing state, pays 0
    _set_headstand(env, asset)
    r = _step(env)
    assert r["hold"] == 1.0
    assert math.isclose(r["orient"], 1.5, rel_tol=1e-5)  # jump capped at max_rate
    assert math.isclose(r["lift"], 2.0, rel_tol=1e-5)
    r = _step(env)
    assert r["orient"] == 0.0 and r["lift"] == 0.0  # frontier reached: nothing to farm


def test_any_other_contact_breaks_the_headstand():
    env, asset, _ = _fake_env()
    _set_headstand(env, asset)
    assert _step(env)["hold"] == 1.0
    env.scene.sensors[hs.NONHEAD_SENSOR].data.found[:] = 1.0
    r = _step(env)
    assert r["hold"] == 0.0


def test_head_velocity_average_in_the_heading_frame():
    """env._hs_ema: (vx, vy, wz) of the head, 0.5 s EMA (the headspin reads wz)."""
    env, asset, _ = _fake_env()
    _set_headstand(env, asset)
    _step(env)
    yaw = float(hs.head_heading(asset.data.body_link_quat_w[:, HEAD])[0])
    asset.data.body_link_lin_vel_w[:, HEAD, 0] = 0.02 * math.cos(yaw)   # forward in the head frame
    asset.data.body_link_lin_vel_w[:, HEAD, 1] = 0.02 * math.sin(yaw)
    asset.data.body_link_ang_vel_w[:, HEAD, 2] = 2.0                     # spinning at 2 rad/s
    _step(env)
    first = env._hs_ema[0].clone()
    for _ in range(200):                                                 # 4 s = 8 time constants
        _step(env)
    last = env._hs_ema[0]
    assert float(first[2]) < 1.0 < 1.98 < float(last[2])                # the EMA has to catch up first
    assert abs(float(last[0]) - 0.02) < 1e-3 and abs(float(last[1])) < 1e-3


# ---- spawns -------------------------------------------------------------------

def _spawn(n=400, **probs):
    torch.manual_seed(0)
    env, asset, _ = _fake_env(n)
    env.sim = SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(n, 7 + 14), qvel=torch.zeros(n, 6 + 14)))
    hs.reset_headstand_state(env, torch.arange(n), **probs)
    return env


def test_headstand_spawn_writes_the_balance_pose():
    env = _spawn(headstand_prob=1.0, standing_prob=0.0, midway_prob=0.0,
                 tilt_max=math.radians(4.0), joint_noise_std=0.0)
    qpos = env.sim.data.qpos
    assert torch.allclose(qpos[:, 2], torch.full_like(qpos[:, 2], hs.HEADSTAND_Z + 0.006))
    assert torch.allclose(qpos[:, 7:], torch.tensor(hs.HEADSTAND_JOINT_POS).expand(len(qpos), -1))
    g_t = np.array(hs.HEADSTAND_GRAVITY_B)
    tilts = []
    yaws = set()
    for q in qpos[:, 3:7].numpy():
        R = _quat_to_mat(q)
        g_b = R.T @ np.array([0.0, 0.0, -1.0])
        tilts.append(math.degrees(math.acos(np.clip(g_b @ g_t / np.linalg.norm(g_t), -1, 1))))
        yaws.add(round(math.atan2(R[1, 0], R[0, 0]), 1))
    assert max(tilts) <= 4.01 and max(tilts) > 2.0
    assert len(yaws) > 20  # random heading
    assert bool(env._hs_fresh.all()) and not bool(env._hs_on.any())


def test_spawn_mix():
    env = _spawn(headstand_prob=0.5, standing_prob=0.5, midway_prob=0.0)
    is_hs = torch.isclose(env.sim.data.qpos[:, 2], torch.tensor(hs.HEADSTAND_Z + 0.006))
    frac = float(is_hs.float().mean())
    assert 0.4 < frac < 0.6
    # the others are roulade standing starts: trunk upright (up-axis z ~ 1)
    q = env.sim.data.qpos[~is_hs, 3:7]
    up_z = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    assert bool((up_z > 0.98).all())


# ---- the balance pose on the plain MuJoCo model ------------------------------------

def test_balance_pose_touches_the_floor_with_the_head_only():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    names = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
             "neck_pitch", "head_pitch", "head_yaw", "head_roll",
             "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
    free = model.joint("trunk_base_freejoint").qposadr[0]
    data.qpos[free:free + 3] = [0.0, 0.0, hs.HEADSTAND_Z - 0.0005]  # 0.5 mm into the floor
    data.qpos[free + 3:free + 7] = hs.HEADSTAND_QUAT
    for name, val in zip(names, hs.HEADSTAND_JOINT_POS):
        j = model.joint(name)
        lo, hi = model.jnt_range[j.id]
        assert lo <= val <= hi, name
        data.qpos[j.qposadr[0]] = val
    mujoco.mj_forward(model, data)
    head = model.body(hs.HEAD_BODY).id
    touching = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if b1 == 0 or b2 == 0:
            touching.add(b2 if b1 == 0 else b1)
    assert touching == {head}
    feet = min(data.xpos[model.body(b).id][2] for b in hs.FOOT_BODIES)
    assert feet > hs.FEET_Z_HIGH
