"""Headstand on the flat top of the head: get into it, then hold it.

Task: Mjlab-HeadStand-Flat-MicroDuck (microduck_headstand_env_cfg.py). It is also
the base of the headspin (headspin_mdp.py), which reuses the state kept here.

The robot starts standing (or partway down, or already balanced), tips forward
with the chin tucked until the flat top of the head is on the floor, lifts its
feet so that ONLY the head touches the ground, and then holds the balance.
Velocities are expressed in the head's own heading frame: in the headstand the
head z-axis is horizontal and points the same way the trunk faces.

Physics check that shaped this task (CPU MuJoCo with the BAM XL330 model the
policies train against): a hand-made COM feedback loop at 50 Hz holds the
head-only balance for 10 s (6/6 from +/-3 deg starts). The neck servos sag
~40 deg under the body, so the policy has to command well past the pose it
wants; static neck torque is 30-40% of stall, peaks ~70%.

Per-env state lives on the env object (created lazily, reset by
reset_headstand_state):
  env._hs_on     - True while only the flat head top touches the ground
  env._hs_ema    - (vx, vy, wz) of the head in the heading frame, 0.5 s EMA
  env._hs_front  - entry-shaping frontiers (orientation, feet lift)
  env._hs_delta  - this step's frontier increments (what the shaping pays)
"""

import math

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp

# ---- balance pose (tools/physics/headstand_constants.py) ---------------------
# COM straight over the centre of the flat head-top face (35 x 81 mm), legs at
# HOME, resting on the floor. Joint order = servo index order.
HEADSTAND_JOINT_POS = (
    0.0, -0.0873, -0.4579, -0.0049, 0.453,       # left leg
    -1.4312, 0.5236, 0.0, 0.0,                   # neck_pitch, head_pitch, head_yaw, head_roll
    0.0, 0.0873, 0.4579, 0.0049, -0.453,         # right leg
)
HEADSTAND_QUAT = (0.81159, -0.00337, 0.58422, 0.0)   # trunk (w, x, y, z), yaw 0: ~71.5 deg forward
HEADSTAND_Z = 0.1301                                  # trunk origin height, head face on the floor
HEADSTAND_GRAVITY_B = (0.9483, 0.0055, -0.3173)       # projected gravity, trunk frame
STAND_GRAVITY_B = (0.0, 0.0, -1.0)

HEAD_BODY = "jaw_soft"
FOOT_BODIES = ("ankle_left", "ankle_right")
HEAD_SENSOR = "head_ground_contact"          # jaw_soft vs terrain (roulade env sensor)
NONHEAD_SENSOR = "nonhead_ground_contact"    # every other robot body vs terrain

# Head-top axis in the jaw_soft frame (same constant as the roulade latch).
HEAD_TOP_AXIS = (0.882, 0.0, 0.471)
TOP_DOWN_MIN = 0.5     # head-top axis world z below -0.5: the TOP of the head is on the floor
TOP_DOWN_FULL = 0.8    # full orientation-shaping credit from here (headstand reads -0.91)

# Ankle-body heights: 0.023 m standing, 0.100 m in the headstand.
FEET_Z_LOW = 0.035
FEET_Z_HIGH = 0.09

EMA_TAU = 0.5          # s, velocity averaging

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ---- small helpers ---------------------------------------------------------------

def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (..., 4) quaternions in (w, x, y, z) order."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def head_top_z(quat: torch.Tensor) -> torch.Tensor:
    """World z of the head-top axis (-1 = the top of the head points at the floor)."""
    w, x, y, z = quat.unbind(-1)
    a, b, c = HEAD_TOP_AXIS
    return (
        2.0 * (x * z - w * y) * a
        + 2.0 * (y * z + w * x) * b
        + (1.0 - 2.0 * (x * x + y * y)) * c
    )


def head_heading(quat: torch.Tensor) -> torch.Tensor:
    """Yaw of the head z-axis projected on the floor: the heading of the command frame."""
    w, x, y, z = quat.unbind(-1)
    zx = 2.0 * (x * z + w * y)
    zy = 2.0 * (y * z - w * x)
    return torch.atan2(zy, zx)


def to_heading_frame(vec_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame (x, y) vectors into the frame whose x-axis has this yaw."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    vx, vy = vec_xy[..., 0], vec_xy[..., 1]
    return torch.stack([c * vx + s * vy, -s * vx + c * vy], dim=-1)


def orientation_potential(gravity_b: torch.Tensor, top_z: torch.Tensor) -> torch.Tensor:
    """0 standing upright, 1 at the headstand orientation with the head top down.

    Trunk orientation progress (angle between projected gravity and the
    headstand's) times a head-top-down factor, so a face-plant with the neck
    at HOME (head top pointing forward, not down) earns nothing.
    """
    g_t = torch.tensor(HEADSTAND_GRAVITY_B, device=gravity_b.device, dtype=gravity_b.dtype)
    g_s = torch.tensor(STAND_GRAVITY_B, device=gravity_b.device, dtype=gravity_b.dtype)
    ang_ref = torch.acos(torch.clamp((g_t * g_s).sum() / torch.linalg.norm(g_t), -1.0, 1.0))
    g = gravity_b / torch.clamp(torch.linalg.norm(gravity_b, dim=-1, keepdim=True), min=1e-6)
    cosang = (g * g_t).sum(-1) / torch.linalg.norm(g_t)
    ang = torch.acos(torch.clamp(cosang, -1.0, 1.0))
    orient = torch.clamp(1.0 - ang / ang_ref, 0.0, 1.0)
    top = torch.clamp(-top_z / TOP_DOWN_FULL, 0.0, 1.0)
    return orient * top


def lift_potential(feet_z: torch.Tensor, head_top_contact: torch.Tensor) -> torch.Tensor:
    """0 with a foot on the floor, 1 with both feet up - only counted on the head top."""
    lift = torch.clamp((feet_z - FEET_Z_LOW) / (FEET_Z_HIGH - FEET_Z_LOW), 0.0, 1.0)
    return lift * head_top_contact.float()


def _any_contact(env, name: str) -> torch.Tensor:
    found = env.scene.sensors[name].data.found
    return (found.view(found.shape[0], -1) > 0).any(dim=-1)


def _hs_buffers(env) -> None:
    if hasattr(env, "_hs_ema"):
        return
    n, dev = env.num_envs, env.device
    env._hs_on = torch.zeros(n, dtype=torch.bool, device=dev)
    env._hs_ema = torch.zeros(n, 3, device=dev)
    env._hs_front = torch.zeros(n, 2, device=dev)
    env._hs_delta = torch.zeros(n, 2, device=dev)
    env._hs_fresh = torch.ones(n, dtype=torch.bool, device=dev)
    env._hs_last_step = -1
    asset = env.scene[_DEFAULT_ASSET_CFG.name]
    env._hs_head_id = asset.find_bodies(HEAD_BODY)[0][0]
    env._hs_feet_ids = [asset.find_bodies(b)[0][0] for b in FOOT_BODIES]


def update_headstand(env) -> None:
    """Refresh the per-env headstand state once per control step."""
    _hs_buffers(env)
    step = int(env.common_step_counter)
    if step == env._hs_last_step:
        return
    env._hs_last_step = step

    asset = env.scene[_DEFAULT_ASSET_CFG.name]
    hq = asset.data.body_link_quat_w[:, env._hs_head_id]
    top_z = torch.nan_to_num(head_top_z(hq), nan=1.0)
    head_c = _any_contact(env, HEAD_SENSOR)
    other_c = _any_contact(env, NONHEAD_SENSOR)
    top_contact = head_c & (top_z < -TOP_DOWN_MIN)
    env._hs_on = top_contact & ~other_c

    # Head velocity in the heading frame, averaged over ~EMA_TAU.
    yaw = head_heading(hq)
    v_xy = to_heading_frame(asset.data.body_link_lin_vel_w[:, env._hs_head_id, :2], yaw)
    wz = asset.data.body_link_ang_vel_w[:, env._hs_head_id, 2:3]
    meas = torch.nan_to_num(torch.cat([v_xy, wz], dim=-1), nan=0.0)
    alpha = min(1.0, float(env.step_dt) / EMA_TAU)
    env._hs_ema = env._hs_ema + alpha * (meas - env._hs_ema)

    # Entry shaping: pay only increments of the best-so-far potentials.
    origin_z = env.scene.terrain.env_origins[:, 2]
    feet_z = torch.stack(
        [asset.data.body_link_pos_w[:, i, 2] for i in env._hs_feet_ids], dim=-1
    ).min(dim=-1).values - origin_z
    pot = torch.stack(
        [
            orientation_potential(asset.data.projected_gravity_b, top_z),
            lift_potential(feet_z, top_contact),
        ],
        dim=-1,
    )
    pot = torch.nan_to_num(pot, nan=0.0)
    front = torch.where(env._hs_fresh.unsqueeze(-1), pot, env._hs_front)
    new_front = torch.maximum(front, pot)
    env._hs_delta = new_front - front
    env._hs_front = new_front
    env._hs_fresh = torch.zeros_like(env._hs_fresh)


# ---- rewards -------------------------------------------------------------------------

def headstand_hold(env) -> torch.Tensor:
    """1 while only the flat top of the head touches the ground, else 0."""
    update_headstand(env)
    return env._hs_on.float()


def headstand_entry_orient(env, max_rate: float = 1.5) -> torch.Tensor:
    """Frontier increments of the orientation potential (tip forward, chin tucked).

    Full entry pays 1.0 in total (times the weight); faster than max_rate
    potential units per second forfeits the excess, so whipping does not pay.
    """
    update_headstand(env)
    dt = float(env.step_dt)
    return torch.clamp(env._hs_delta[:, 0], max=max_rate * dt) / dt


def headstand_entry_lift(env, max_rate: float = 2.0) -> torch.Tensor:
    """Frontier increments of the feet-lift potential (only on the head top)."""
    update_headstand(env)
    dt = float(env.step_dt)
    return torch.clamp(env._hs_delta[:, 1], max=max_rate * dt) / dt


def neck_torque_l2(env, actuator_ids: tuple = (5, 6)) -> torch.Tensor:
    """Sum of squared neck/head-pitch actuator torques (positive; negative weight).

    Holding the headstand loads these two servos continuously: this is heat
    pressure toward lower-torque balance poses, introduced late by curriculum.
    """
    asset = env.scene[_DEFAULT_ASSET_CFG.name]
    tau = asset.data.actuator_force[:, list(actuator_ids)]
    return torch.nan_to_num(tau, nan=0.0).pow(2).sum(dim=-1)


# ---- reset ------------------------------------------------------------------------

def reset_headstand_state(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    headstand_prob: float = 0.5,
    standing_prob: float = 0.3,
    midway_prob: float = 0.2,
    tilt_max: float = math.radians(4.0),
    joint_noise_std: float = 0.05,
    z_gap: float = 0.006,
    roulade_params: dict | None = None,
):
    """Three spawn buckets (reverse curriculum):

    * headstand: the balance pose with random yaw, up to tilt_max of tilt and
      joint noise, dropped from z_gap above the floor (teaches holding; the
      ones that start to fall teach recovery from the head tripod).
    * standing / midway: reset_roulade_state's standing start and its
      "mid-roll" start restricted to the first part of a forward roll (head
      coming down to the floor, chin tucked) - teaches the entry.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    n = len(env_ids)
    total = max(headstand_prob + standing_prob + midway_prob, 1e-6)
    is_hs = torch.rand(n, device=env.device) < (headstand_prob / total)

    rest = env_ids[~is_hs]
    if len(rest) > 0:
        microduck_mdp.reset_roulade_state(
            env, rest, asset_cfg=asset_cfg,
            standing_prob=standing_prob, midroll_prob=midway_prob,
            **dict(roulade_params or {}),
        )

    hs_ids = env_ids[is_hs]
    m = len(hs_ids)
    if m > 0:
        asset = env.scene[asset_cfg.name]
        dev = env.device
        yaw = torch.rand(m, device=dev) * 2.0 * math.pi - math.pi
        zeros = torch.zeros(m, device=dev)
        q_yaw = torch.stack([torch.cos(yaw * 0.5), zeros, zeros, torch.sin(yaw * 0.5)], dim=-1)
        phi = torch.rand(m, device=dev) * 2.0 * math.pi
        ang = torch.rand(m, device=dev) * tilt_max
        s = torch.sin(ang * 0.5)
        q_tilt = torch.stack([torch.cos(ang * 0.5), s * torch.cos(phi), s * torch.sin(phi), zeros], dim=-1)
        q_hs = torch.tensor(HEADSTAND_QUAT, device=dev).expand(m, 4)
        quat = quat_mul(q_yaw, quat_mul(q_tilt, q_hs))
        quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)

        env.sim.data.qpos[hs_ids, 2] = HEADSTAND_Z + z_gap
        env.sim.data.qpos[hs_ids, 3:7] = quat
        env.sim.data.qvel[hs_ids] = 0.0
        servo_ids = microduck_mdp._servo_joint_ids(env, asset)
        cols = torch.tensor([7 + servo_ids[j] for j in range(len(HEADSTAND_JOINT_POS))],
                            device=dev, dtype=torch.long)
        pose = torch.tensor(HEADSTAND_JOINT_POS, device=dev).expand(m, -1)
        if joint_noise_std > 0.0:
            pose = pose + torch.randn(m, len(HEADSTAND_JOINT_POS), device=dev) * joint_noise_std
        env.sim.data.qpos[hs_ids.unsqueeze(1), cols.unsqueeze(0)] = pose

    _hs_buffers(env)
    env._hs_on[env_ids] = False
    env._hs_ema[env_ids] = 0.0
    env._hs_front[env_ids] = 0.0
    env._hs_delta[env_ids] = 0.0
    env._hs_fresh[env_ids] = True
