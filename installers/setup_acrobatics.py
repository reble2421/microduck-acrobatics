#!/usr/bin/env python3
"""Add the microduck-acrobatics tasks to a fresh microduck_rl clone.

    git clone https://github.com/pollen-robotics/microduck_rl && cd microduck_rl
    git checkout 53b8971b61baf5b7f3c16d135dd7cac37623de4b
    python3 /path/to/setup_acrobatics.py

Tasks: Mjlab-HeadStand-Flat-MicroDuck, Mjlab-BackRoll-Flat-MicroDuck,
Mjlab-HeadSpin-Flat-MicroDuck, Mjlab-BackRollRobust-Flat-MicroDuck,
Mjlab-HeadSpinRobust-Flat-MicroDuck, Mjlab-HeadSpinFast5-Flat-MicroDuck,
Mjlab-HeadSpinFast6-Flat-MicroDuck.

Everything is checked first; nothing is changed unless every check passes. Running it
again is safe (it only fills in what is missing). Two existing files of the clone are
changed: tasks/__init__.py gets the registrations appended, and tasks/mdp.py gets nine
small edits that let the roulade's reset and head latch run backwards (a roll
direction and a head-latch window per task; the defaults keep the forward roulade
exactly as it was).
"""

import hashlib
import subprocess
import sys
from pathlib import Path

PINNED = "53b8971b61baf5b7f3c16d135dd7cac37623de4b"
TAG = "[acrobatics]"
PAYLOAD_SHA256 = "bf37370d7032d7bf0efd034e6155e2366d4c931c2ea19bf5e4cc06b0b8a887cb"

TEXT_00 = r'''"""Headstand on the flat top of the head: get into it, then hold it.

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
'''

TEXT_01 = r'''"""Microduck headstand: stand on the flat top of the head and hold the balance.

Episodic-to-perpetual trick. The robot starts standing, tips forward with the
chin tucked (the first ~70 deg of a roulade), lifts both feet so that ONLY the
head touches the floor, and then holds the balance. It is the base of the
headspin task (microduck_headspin_env_cfg.py).

Built on the roulade env (same robot model with head collisions, DR, 61-D
observation layout, sensors, NaN guard, regularisers). What changes:

  * spawns (reset_headstand_state): balanced headstand / standing / head coming
    down (roulade mid-roll spawns restricted to 40-100 deg), reverse curriculum
    shifting toward standing starts as training goes on.
  * task rewards (headstand_mdp.py), all state-gated:
      - headstand_hold        1 while only the flat head top touches the floor
      - entry_orient / lift   frontier (best-so-far) shaping: tip forward with
                              the head top down, then lift the feet. Pays once,
                              rate-capped, so neither camping nor whipping pays.
  * the roulade's roll terms, landing terms and arrival damper are removed.
  * the twist command stays in the observation (the shared 61-D layout) with
    zero ranges: hold still. The headspin task fills it with spin commands.
  * a late neck-torque penalty (the neck servos hold ~60% of the body weight).
  * episodes are 12 s: ~2 s to get up, the rest to balance.

Physics check before training: head-only balance is controllable at 50 Hz
through the BAM XL330 model (a crude COM loop holds it 10 s). The top-of-head
face is small (35 x 81 mm under a COM 97 mm up), so this is a hard balance task.
"""

import math
from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.tasks import headstand_mdp as hs
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    TUCK_OVERRIDES,
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)

EPISODE_LENGTH_S = 12.0

# The twist command is part of the shared observation but asks for nothing here:
# zero ranges, hold still (the headspin task sets its own ranges and stages).
COMMAND_STAGES = [
    {"step": 0, "lin_vel_range": 0.0, "ang_vel_range": 0.0},
]

SPAWN_STAGES = [
    {"step": 0,          "params": {"headstand_prob": 0.50, "standing_prob": 0.30, "midway_prob": 0.20}},
    {"step": 3000 * 24,  "params": {"headstand_prob": 0.35, "standing_prob": 0.45, "midway_prob": 0.20}},
    {"step": 6000 * 24,  "params": {"headstand_prob": 0.25, "standing_prob": 0.60, "midway_prob": 0.15}},
]

# Roulade terms that are about rolling / landing, not about the headstand.
ROULADE_ONLY_REWARDS = (
    "roulade_progress", "roulade_overspeed", "roulade_head_pivot",
    "roulade_landing_composite", "roulade_upright_after_roll",
    "roulade_height_after_roll", "roulade_landing_sharp", "roulade_stand_tax",
    "roulade_rise_velocity", "roulade_sagittal", "roulade_lateral_vel",
    "roulade_flatness", "arrival_damping",
)
ROULADE_ONLY_CURRICULA = ("roulade_spawn_mix", "arrival_damping_weight")

# "Midway" spawns: roulade mid-roll spawns limited to the head coming down.
MIDWAY_PARAMS = {
    "standing_z_min":      0.11,
    "standing_z_max":      0.12,
    "standing_tilt_max":   math.radians(5.0),
    "forward_vel_range":   (0.0, 0.0),
    "midroll_pitch_min":   math.radians(40.0),
    "midroll_pitch_max":   math.radians(100.0),
    "midroll_z_min":       0.05,
    "midroll_z_max":       0.10,
    "midroll_omega_range": (0.0, 1.0),
    "tuck_overrides":      TUCK_OVERRIDES,
    "tuck_factor_range":   (0.5, 1.0),
    "joint_noise_std":     0.08,
}


def make_microduck_headstand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_roulade_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # Every robot body except the head, against the floor: the headstand gate.
    nonhead_ground_cfg = ContactSensorCfg(
        name=hs.NONHEAD_SENSOR,
        primary=ContactMatch(mode="body", pattern=".*", entity="robot", exclude=(hs.HEAD_BODY,)),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (nonhead_ground_cfg,)

    # ---- rewards ---------------------------------------------------------------
    for name in ROULADE_ONLY_REWARDS:
        cfg.rewards.pop(name, None)

    cfg.rewards["headstand_hold"] = RewardTermCfg(func=hs.headstand_hold, weight=1.0)
    cfg.rewards["headstand_entry_orient"] = RewardTermCfg(
        func=hs.headstand_entry_orient, weight=2.0, params={"max_rate": 1.5},
    )
    cfg.rewards["headstand_entry_lift"] = RewardTermCfg(
        func=hs.headstand_entry_lift, weight=2.0, params={"max_rate": 2.0},
    )
    cfg.rewards["neck_torque"] = RewardTermCfg(
        func=hs.neck_torque_l2, weight=0.0, params={"actuator_ids": (5, 6)},
    )

    # ---- command -------------------------------------------------------------
    cmd = cfg.commands["twist"]
    first = COMMAND_STAGES[0]
    cmd.ranges.lin_vel_x = (-first["lin_vel_range"], first["lin_vel_range"])
    cmd.ranges.lin_vel_y = (-first["lin_vel_range"], first["lin_vel_range"])
    cmd.ranges.ang_vel_z = (-first["ang_vel_range"], first["ang_vel_range"])
    cmd.ranges.heading = None
    cmd.heading_command = False
    cmd.rel_heading_envs = 0.0
    cmd.rel_standing_envs = 0.25     # exact-zero command: "hold still" must be trained
    cmd.rel_forward_envs = 0.0       # the base default forces vx >= 0.3 m/s there
    cmd.rel_world_envs = 0.0
    cmd.init_velocity_prob = 0.0
    cmd.rel_turn_in_place_envs = 0.1
    cmd.resampling_time_range = (3.0, 6.0)
    cmd.debug_vis = False

    # ---- events ----------------------------------------------------------------
    cfg.events.pop("set_roulade_state", None)
    cfg.events["set_headstand_state"] = EventTermCfg(
        func=hs.reset_headstand_state,
        mode="reset",
        params={
            **SPAWN_STAGES[0]["params"],
            "tilt_max": math.radians(4.0),
            "joint_noise_std": 0.05,
            "z_gap": 0.006,
            "roulade_params": dict(MIDWAY_PARAMS),
        },
    )

    # ---- curriculum ------------------------------------------------------------
    for name in ROULADE_ONLY_CURRICULA:
        cfg.curriculum.pop(name, None)
    cfg.curriculum["headstand_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={"event_name": "set_headstand_state", "param_stages": deepcopy(SPAWN_STAGES)},
    )
    cfg.curriculum["headstand_command_ranges"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_command_ranges_curriculum,
        params={"command_name": "twist", "velocity_stages": deepcopy(COMMAND_STAGES)},
    )
    # Balance needs quick corrections: smoothing ramps a little gentler than
    # the roulade's (-0.1 / -0.2 / -0.4).
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0,          "weight": -0.1},
        {"step": 2000 * 24,  "weight": -0.2},
        {"step": 4000 * 24,  "weight": -0.3},
    ]
    cfg.curriculum["neck_torque_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "neck_torque",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 3000 * 24,  "weight": -0.1},
                {"step": 6000 * 24,  "weight": -0.2},
            ],
        },
    )
    return cfg


MicroduckHeadstandRlCfg = replace(
    deepcopy(MicroduckRouladeRlCfg),
    algorithm=replace(deepcopy(MicroduckRouladeRlCfg.algorithm), symmetry_cfg=None),
    experiment_name="microduck_headstand",
    run_name="microduck_headstand",
)
'''

TEXT_02 = r'''"""Microduck backward roll ("back roll") - the roulade, run the other way.

Built on the roulade env (microduck_roulade_env_cfg.py): same robot model, DR,
observations, tuck, rewards and reverse curriculum. Only the roll direction and
the two direction-dependent progress windows change. A backward roll at
progress p passes through the same body orientations as a forward roll at
360 deg - p, so both windows are mirrored:

  * set_roulade_state: direction = -1 -> mid-roll spawns pitched BACKWARD with
    backward angular momentum; the accumulator counts backward rotation.
  * head-top latch window 190..340 deg (roulade: 20..170). Measured on the rigid
    roulade tuck: the head-top-down condition holds at 210..330 deg of a
    backward roll, the mirror of 30..150 deg forward.
  * head-pivot shaping window 120..330 deg (roulade: 30..240).

Progress frontier, completion gates (260/330 deg), flatness / sagittal /
overspeed terms and the landing composite are direction-agnostic.

Physics check before training (CPU MuJoCo, rigid tuck): the COM climb over the
head is the same 69 mm (0.50 J) as in the forward roll, but a backward roll
climbs it with the legs in the air - only the neck pushes on the floor. The
static neck torque needed peaks at ~46% of XL330 stall (head_pitch): feasible,
with about a 2x margin.
"""

import math
from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)

BACKROLL_DIRECTION = -1.0
BACKROLL_HEAD_LATCH = (math.radians(190.0), math.radians(340.0))
BACKROLL_HEAD_PIVOT = (math.radians(120.0), math.radians(330.0))


def make_microduck_backroll_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_roulade_env_cfg(play=play)

    reset = cfg.events["set_roulade_state"].params
    reset["direction"] = BACKROLL_DIRECTION
    reset["head_latch_lo"], reset["head_latch_hi"] = BACKROLL_HEAD_LATCH

    pivot = cfg.rewards["roulade_head_pivot"].params
    pivot["angle_lo"], pivot["angle_hi"] = BACKROLL_HEAD_PIVOT
    return cfg


MicroduckBackrollRlCfg = replace(
    deepcopy(MicroduckRouladeRlCfg),
    experiment_name="microduck_backroll",
    run_name="microduck_backroll",
)
'''

TEXT_03 = r'''"""Headspin: spin on the top of the head at the commanded rate, stop on command.

Task: Mjlab-HeadSpin-Flat-MicroDuck (microduck_headspin_env_cfg.py), built on the
headstand task (same entry shaping, gate, spawns and hold reward; headstand_mdp.py).

State kept on the env object (created lazily, refreshed once per control step):
  env._hs_on, env._hs_ema[:, 2]  headstand gate and head yaw rate, 0.5 s EMA (headstand_mdp)
  env._hsp_com_v                 whole-robot COM velocity, 0.5 s EMA, heading frame
  env._hsp_sup_v                 velocity of the head-top face centre (the support
                                 patch on the floor), 0.5 s EMA, heading frame

Rewards (omega = yaw rate EMA, c = commanded yaw rate):
  headspin_track      1 - |omega - c| / |c|, clipped to [floor, 1], for a spin command.
                      The positive part is gated on the COM staying over the support
                      (its EMA speed below drift_ref): spinning while wandering does
                      not pay. Penalties are never gated.
  spin_when_stopped   |omega| / spin_ref (clip 1) for a stop command: a cost, so
                      spinning down on command is trained.
  still_when_stopped  exp(-|v_support|^2 / lin_std^2 - omega^2 / ang_std^2) for a stop
                      command: stay put on the head.

Commands (HeadSpinCommand): zero_frac exact zero (stop), the rest a pure yaw-rate
command with |wz| in [min_frac, 1] x range and a random sign; no linear command.
"""

import math
from dataclasses import dataclass

import torch

from mjlab_microduck.tasks import headstand_mdp as hs
from mjlab_microduck.tasks import mdp as microduck_mdp

LIN_EPS = 1e-3      # m/s: a linear command below this counts as zero
ANG_EPS = 0.02      # rad/s: a yaw command below this counts as zero
EMA_TAU = 0.5       # s, same averaging as the headstand state
SPIN_REF = 0.4      # rad/s: yaw rate that costs the full spin_when_stopped penalty
DRIFT_REF = 0.03    # m/s: COM speed (0.5 s EMA) that removes all spin pay
# Centre of the flat head-top face in the jaw_soft (head) frame; it sits on the
# floor in the balanced headstand (checked in CPU MuJoCo: z = -1 mm).
FACE_CENTER_HEAD = (0.0407513645315408, 0.0009440486873391888, -0.07720818445096261)


def _split_gate(r: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Gate the pay, never the penalty."""
    return torch.clamp(r, min=0.0) * gate + torch.clamp(r, max=0.0)


# ---- whole-robot COM velocity ------------------------------------------------------

def _com_buffers(env) -> None:
    if hasattr(env, "_hsp_com_ema_w"):
        return
    n, dev = env.num_envs, env.device
    env._hsp_com_prev = torch.zeros(n, 3, device=dev)
    env._hsp_com_ema_w = torch.zeros(n, 2, device=dev)
    env._hsp_com_v = torch.zeros(n, 2, device=dev)       # EMA COM velocity, heading frame
    env._hsp_com_fresh = torch.ones(n, dtype=torch.bool, device=dev)
    env._hsp_com_last_step = -1


def robot_com(env) -> torch.Tensor:
    """(N, 3) world position of the whole-robot COM (includes mass DR)."""
    asset = env.scene["robot"]
    return env.sim.data.subtree_com[:, asset.indexing.root_body_id]


def update_com_velocity(env) -> None:
    """Refresh the headstand state and the COM velocity EMA once per control step."""
    hs.update_headstand(env)            # env._hs_on, env._hs_ema[:, 2] (yaw rate EMA)
    _com_buffers(env)
    step = int(env.common_step_counter)
    if step == env._hsp_com_last_step:
        return
    env._hsp_com_last_step = step
    com = torch.nan_to_num(robot_com(env).clone(), nan=0.0)
    fresh = env._hsp_com_fresh.unsqueeze(-1)
    prev = torch.where(fresh, com, env._hsp_com_prev)
    v = (com[:, :2] - prev[:, :2]) / float(env.step_dt)
    alpha = min(1.0, float(env.step_dt) / EMA_TAU)
    ema = env._hsp_com_ema_w + alpha * (v - env._hsp_com_ema_w)
    env._hsp_com_ema_w = torch.where(fresh, torch.zeros_like(ema), ema)
    env._hsp_com_prev = com
    env._hsp_com_fresh = torch.zeros_like(env._hsp_com_fresh)
    asset = env.scene["robot"]
    hq = asset.data.body_link_quat_w[:, env._hs_head_id]
    yaw = torch.nan_to_num(hs.head_heading(hq), nan=0.0)
    env._hsp_com_v = hs.to_heading_frame(env._hsp_com_ema_w, yaw)


def reset_com_velocity(env, env_ids: torch.Tensor) -> None:
    """Reset event: the next update starts the COM difference afresh (no spike)."""
    _com_buffers(env)
    if env_ids is None:
        env._hsp_com_fresh[:] = True
    else:
        env._hsp_com_fresh[env_ids.to(env.device, dtype=torch.long)] = True


# ---- velocity of the support patch (head-top face centre) ------------------------------

def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (N, 3) by quaternions q (N, 4) in (w, x, y, z) order."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _support_buffers(env) -> None:
    if hasattr(env, "_hsp_sup_ema_w"):
        return
    n, dev = env.num_envs, env.device
    env._hsp_sup_ema_w = torch.zeros(n, 2, device=dev)
    env._hsp_sup_v = torch.zeros(n, 2, device=dev)        # support velocity EMA, heading frame
    env._hsp_sup_fresh = torch.ones(n, dtype=torch.bool, device=dev)
    env._hsp_sup_last_step = -1


def support_velocity_w(env) -> torch.Tensor:
    """(N, 3) world velocity of the head-top face centre."""
    asset = env.scene["robot"]
    hid = env._hs_head_id
    q = asset.data.body_link_quat_w[:, hid]
    p = torch.tensor(FACE_CENTER_HEAD, device=env.device, dtype=q.dtype).expand(q.shape[0], 3)
    r = _quat_apply(q, p)
    return asset.data.body_link_lin_vel_w[:, hid] + torch.cross(asset.data.body_link_ang_vel_w[:, hid], r, dim=-1)


def update_support_velocity(env) -> None:
    """Refresh the headstand state and the support velocity EMA once per control step."""
    hs.update_headstand(env)            # env._hs_on, env._hs_ema[:, 2] (yaw rate), env._hs_head_id
    _support_buffers(env)
    step = int(env.common_step_counter)
    if step == env._hsp_sup_last_step:
        return
    env._hsp_sup_last_step = step
    v = torch.nan_to_num(support_velocity_w(env)[:, :2], nan=0.0)
    alpha = min(1.0, float(env.step_dt) / EMA_TAU)
    fresh = env._hsp_sup_fresh.unsqueeze(-1)
    prev = torch.where(fresh, torch.zeros_like(v), env._hsp_sup_ema_w)
    env._hsp_sup_ema_w = prev + alpha * (v - prev)
    env._hsp_sup_fresh = torch.zeros_like(env._hsp_sup_fresh)
    asset = env.scene["robot"]
    yaw = torch.nan_to_num(hs.head_heading(asset.data.body_link_quat_w[:, env._hs_head_id]), nan=0.0)
    env._hsp_sup_v = hs.to_heading_frame(env._hsp_sup_ema_w, yaw)


def reset_support_velocity(env, env_ids: torch.Tensor) -> None:
    """Reset event: start the support velocity EMA afresh."""
    _support_buffers(env)
    if env_ids is None:
        env._hsp_sup_fresh[:] = True
    else:
        env._hsp_sup_fresh[env_ids.to(env.device, dtype=torch.long)] = True


# ---- rewards ----------------------------------------------------------------------

def headspin_track(env, command_name: str = "twist", floor: float = -0.25,
                   drift_ref: float = DRIFT_REF) -> torch.Tensor:
    """1 - |wz - c| / |c| for a spin command, pay gated on the COM staying over the support."""
    update_com_velocity(env)
    wz = env.command_manager.get_command(command_name)[:, 2]
    mag = torch.abs(wz)
    err = torch.abs(env._hs_ema[:, 2] - wz) / torch.clamp(mag, min=ANG_EPS)
    r = torch.clamp(1.0 - err, min=floor, max=1.0)
    gate = torch.clamp(1.0 - torch.linalg.norm(env._hsp_com_v, dim=-1) / drift_ref, min=0.0, max=1.0)
    return env._hs_on.float() * (mag > ANG_EPS).float() * _split_gate(r, gate)


def spin_when_stopped(env, command_name: str = "twist", spin_ref: float = SPIN_REF) -> torch.Tensor:
    """|wz| / spin_ref clipped to [0, 1] when no spin is commanded.
    A cost (>= 0): give it a NEGATIVE weight."""
    update_com_velocity(env)
    c = env.command_manager.get_command(command_name)
    no_turn = torch.abs(c[:, 2]) <= ANG_EPS
    cost = torch.clamp(torch.abs(env._hs_ema[:, 2]) / spin_ref, min=0.0, max=1.0)
    return env._hs_on.float() * no_turn.float() * cost


def still_when_stopped(env, command_name: str = "twist", lin_std: float = 0.004,
                       ang_std: float = 0.1) -> torch.Tensor:
    """Stay put for an exact-zero command: support speed and yaw rate near zero."""
    update_support_velocity(env)
    c = env.command_manager.get_command(command_name)
    zero = (torch.linalg.norm(c[:, :2], dim=-1) <= LIN_EPS) & (torch.abs(c[:, 2]) <= ANG_EPS)
    err = (env._hsp_sup_v ** 2).sum(dim=-1) / (lin_std * lin_std) + env._hs_ema[:, 2] ** 2 / (ang_std * ang_std)
    return env._hs_on.float() * zero.float() * torch.exp(-err)


# ---- commands ---------------------------------------------------------------------

def sample_spin_rates(n: int, ang_max: float, min_frac: float, device) -> torch.Tensor:
    """(n,) yaw-rate commands: |wz| uniform in [min_frac, 1] x ang_max, random sign."""
    mag = (min_frac + (1.0 - min_frac) * torch.rand(n, device=device)) * ang_max
    sign = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    return sign * mag


class HeadSpinCommand(microduck_mdp.VelocityCommandCommandOnly):
    """Two buckets per resample: exact zero (zero_frac) or a pure spin command."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        zero = torch.rand(n, device=self.device) < self.cfg.zero_frac
        ang_max = max(abs(self.cfg.ranges.ang_vel_z[0]), abs(self.cfg.ranges.ang_vel_z[1]))
        self.vel_command_b[env_ids, 0:2] = 0.0
        self.vel_command_b[env_ids, 2] = sample_spin_rates(n, ang_max, self.cfg.min_frac, self.device)
        self.is_standing_env[env_ids] = zero          # standing envs get an exact-zero command
        self.vel_command_b[env_ids[zero]] = 0.0
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]


@dataclass(kw_only=True)
class HeadSpinCommandCfg(microduck_mdp.VelocityCommandCommandOnlyCfg):
    zero_frac: float = 0.25
    min_frac: float = 0.5

    def build(self, env) -> "HeadSpinCommand":
        return HeadSpinCommand(self, env)


def deg_s(rad_s: float) -> float:
    return math.degrees(rad_s)
'''

TEXT_04 = r'''"""Microduck headspin: spin on the head top at a commanded rate, stop on command.

Built on Mjlab-HeadStand-Flat-MicroDuck (same robot, DR, observation, spawns,
entry shaping and hold reward). What changes (reasons in headspin_mdp.py):

  * rewards added, in this order:
      spin_when_stopped   |wz| / 0.4 rad/s (clip 1) for a stop command     weight -1.0
      still_when_stopped  stay put on the head for a stop command          weight  1.0
      headspin_track      1 - |wz - c| / |c|, pay gated on the COM staying  weight  2.0
  * two reset events start the COM / support velocity averages afresh.
  * commands (HeadSpinCommand): 25 % exact zero (stop), 75 % pure spin with |wz| at
    least half the current range and a random sign; no linear command. The range
    grows 1.0 -> 2.0 -> 3.0 rad/s (57 -> 115 -> 172 deg/s) at 800 / 1600 iterations.
    A run resumed from a checkpoint restores the step counter, so a fine-tune from a
    long headstand run starts at the last stage (1.5-3 rad/s).
  * the smoothing (action rate) and neck-torque weights stay at their first stage
    (-0.1 and 0): the spin has to be found before the motion is taxed.

Same experiment folder ("microduck_headstand") so it can be fine-tuned from a
headstand run with --agent.resume True --agent.load-run ... --agent.load-checkpoint ...
"""

from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, RewardTermCfg

from mjlab_microduck.tasks import headspin_mdp as hsp
from mjlab_microduck.tasks.microduck_headstand_env_cfg import (
    MicroduckHeadstandRlCfg,
    make_microduck_headstand_env_cfg,
)

COMMAND_STAGES = [
    {"step": 0,          "lin_vel_range": 0.0, "ang_vel_range": 1.0},
    {"step": 800 * 24,   "lin_vel_range": 0.0, "ang_vel_range": 2.0},
    {"step": 1600 * 24,  "lin_vel_range": 0.0, "ang_vel_range": 3.0},
]
ZERO_FRAC = 0.25            # stop commands (exact zero)
MIN_FRAC = 0.5              # |wz| at least this fraction of the current range
STANDING_ENVS = 0.20        # base command class buckets, before HeadSpinCommand overrides them
TURN_IN_PLACE_ENVS = 0.0


def make_microduck_headspin_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_headstand_env_cfg(play=play)

    # ---- rewards ---------------------------------------------------------------
    cfg.rewards["spin_when_stopped"] = RewardTermCfg(
        func=hsp.spin_when_stopped, weight=-1.0,
        params={"command_name": "twist", "spin_ref": hsp.SPIN_REF},
    )
    cfg.rewards["still_when_stopped"] = RewardTermCfg(
        func=hsp.still_when_stopped, weight=1.0,
        params={"command_name": "twist", "lin_std": 0.004, "ang_std": 0.1},
    )
    cfg.rewards["headspin_track"] = RewardTermCfg(
        func=hsp.headspin_track, weight=2.0,
        params={"command_name": "twist", "floor": -0.25, "drift_ref": hsp.DRIFT_REF},
    )

    # ---- events ----------------------------------------------------------------
    cfg.events["reset_com_velocity"] = EventTermCfg(func=hsp.reset_com_velocity, mode="reset")
    cfg.events["reset_support_velocity"] = EventTermCfg(func=hsp.reset_support_velocity, mode="reset")

    # ---- command ---------------------------------------------------------------
    cmd = cfg.commands["twist"]
    first = COMMAND_STAGES[0]
    cmd.ranges.lin_vel_x = (0.0, 0.0)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (-first["ang_vel_range"], first["ang_vel_range"])
    cmd.rel_standing_envs = STANDING_ENVS
    cmd.rel_turn_in_place_envs = TURN_IN_PLACE_ENVS     # HeadSpinCommand does its own buckets
    cfg.commands["twist"] = hsp.HeadSpinCommandCfg(**vars(cmd), zero_frac=ZERO_FRAC, min_frac=MIN_FRAC)

    # ---- curriculum ------------------------------------------------------------
    cfg.curriculum["headstand_command_ranges"].params["velocity_stages"] = deepcopy(COMMAND_STAGES)
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [{"step": 0, "weight": -0.1}]
    cfg.curriculum["neck_torque_weight"].params["weight_stages"] = [{"step": 0, "weight": 0.0}]
    return cfg


MicroduckHeadSpinRlCfg = replace(
    deepcopy(MicroduckHeadstandRlCfg),
    experiment_name="microduck_headstand",
    run_name="microduck_headspin",
)
'''

TEXT_05 = r'''"""Latency-robust versions of the backroll and the headspin.

Why (bench, 2026-10-02): both tricks are perfect when the motor commands arrive 15-30 ms
late, as in training (BAM actuator delay_min_lag=3 / delay_max_lag=6 physics steps),
and break without that delay - the condition scripts/infer_policy.py rehearses with:
  backroll  clean 50/50 with the training delay, 17/50 without (CPU MuJoCo + BAM)
  headspin  0.73-0.92 of the commanded rate with the delay, 0.08-0.35 without
The published roulade is clean either way. Both policies learned motions timed for a
latency they always had. Here the delay is randomised over 0-6 physics steps (0-30 ms)
so they must work across it.

  Mjlab-BackRollRobust-Flat-MicroDuck  backroll, command delay 0-6 physics steps
  Mjlab-HeadSpinRobust-Flat-MicroDuck  headspin, command delay 0-6 physics steps, and
                                       spin commands from 1/3 of the range (1-3 rad/s at
                                       the last stage: the first headspin run had
                                       forgotten slow spins, its commands were 1.5-3)
Same experiment folders as the base tasks, so both are fine-tuned from their runs.
"""

from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_microduck.tasks.microduck_backroll_env_cfg import (
    MicroduckBackrollRlCfg,
    make_microduck_backroll_env_cfg,
)
from mjlab_microduck.tasks.microduck_headspin_env_cfg import (
    MicroduckHeadSpinRlCfg,
    make_microduck_headspin_env_cfg,
)

DELAY_RANGE = (0, 6)          # physics steps of 5 ms; the base tasks use (3, 6)
HEADSPIN_MIN_FRAC = 1.0 / 3.0


def widen_command_delay(cfg: ManagerBasedRlEnvCfg, lo: int = DELAY_RANGE[0],
                        hi: int = DELAY_RANGE[1]) -> ManagerBasedRlEnvCfg:
    """Randomise the actuator command delay over [lo, hi] physics steps (copies the robot cfg)."""
    robot = deepcopy(cfg.scene.entities["robot"])
    acts = tuple(replace(a, delay_min_lag=lo, delay_max_lag=hi) for a in robot.articulation.actuators)
    robot.articulation = replace(robot.articulation, actuators=acts)
    cfg.scene.entities["robot"] = robot
    return cfg


def make_microduck_backroll_robust_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return widen_command_delay(make_microduck_backroll_env_cfg(play=play))


def make_microduck_headspin_robust_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = widen_command_delay(make_microduck_headspin_env_cfg(play=play))
    cfg.commands["twist"].min_frac = HEADSPIN_MIN_FRAC
    return cfg


MicroduckBackrollRobustRlCfg = replace(
    deepcopy(MicroduckBackrollRlCfg), run_name="microduck_backroll_robust",
)
MicroduckHeadSpinRobustRlCfg = replace(
    deepcopy(MicroduckHeadSpinRlCfg), run_name="microduck_headspin_robust",
)
'''

TEXT_06 = r'''"""Fast headspin: the latency-robust headspin with spin commands up to 5 or 6 rad/s.

Why (bench, 2026-10-08): the latency-robust headspin (trained on 1-3 rad/s) already
follows faster commands it never saw - 4 rad/s at 1.05x, 5 rad/s at 0.92x - but falls
behind at 6 rad/s (0.61-0.76x) and starts to drop off its head. Faster spins read better
on video, so the command range is widened, on command (stop included), not just faster.

  Mjlab-HeadSpinFast6-Flat-MicroDuck  |wz| commands 1-6 rad/s
  Mjlab-HeadSpinFast5-Flat-MicroDuck  |wz| commands 1-5 rad/s

Everything else is the latency-robust headspin (command delay 0-6 physics steps, same
rewards, 25 % stop commands). The range is set directly, without stages: the run is
fine-tuned from a policy that already spins at 4-5 rad/s, and a resumed run restores
the step counter, so stage steps would be read against the old run's count.
Same experiment folder as the headspin tasks, so it fine-tunes from their runs.
"""

from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_microduck.tasks.microduck_robust_env_cfg import (
    MicroduckHeadSpinRobustRlCfg,
    make_microduck_headspin_robust_env_cfg,
)

MIN_RAD_S = 1.0     # slowest spin command, as in the robust task


def make_microduck_headspin_fast_env_cfg(play: bool = False, ang_max: float = 6.0) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_headspin_robust_env_cfg(play=play)
    cfg.curriculum["headstand_command_ranges"].params["velocity_stages"] = [
        {"step": 0, "lin_vel_range": 0.0, "ang_vel_range": ang_max},
    ]
    cmd = cfg.commands["twist"]
    cmd.ranges.ang_vel_z = (-ang_max, ang_max)
    cmd.min_frac = MIN_RAD_S / ang_max
    return cfg


def make_microduck_headspin_fast6_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return make_microduck_headspin_fast_env_cfg(play=play, ang_max=6.0)


def make_microduck_headspin_fast5_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return make_microduck_headspin_fast_env_cfg(play=play, ang_max=5.0)


MicroduckHeadSpinFast6RlCfg = replace(
    deepcopy(MicroduckHeadSpinRobustRlCfg), run_name="microduck_headspin_fast6",
)
MicroduckHeadSpinFast5RlCfg = replace(
    deepcopy(MicroduckHeadSpinRobustRlCfg), run_name="microduck_headspin_fast5",
)
'''

TEXT_07 = r'''"""Headstand task (Mjlab-HeadStand-Flat-MicroDuck): config invariants, the
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
'''

TEXT_08 = r'''"""Backward roll (Mjlab-BackRoll-Flat-MicroDuck): the roulade with the roll
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
'''

TEXT_09 = r'''"""Headspin (Mjlab-HeadSpin-Flat-MicroDuck): config invariants, the spin command sampler,
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
'''

TEXT_10 = r'''"""Latency-robust backroll / headspin: the delay range is widened, nothing else moves,
and the base tasks are untouched. CPU only, no warp env is built."""

from mjlab_microduck.tasks.microduck_backroll_env_cfg import MicroduckBackrollRlCfg, make_microduck_backroll_env_cfg
from mjlab_microduck.tasks.microduck_headspin_env_cfg import MicroduckHeadSpinRlCfg, make_microduck_headspin_env_cfg
from mjlab_microduck.tasks.microduck_robust_env_cfg import (
    DELAY_RANGE,
    HEADSPIN_MIN_FRAC,
    MicroduckBackrollRobustRlCfg,
    MicroduckHeadSpinRobustRlCfg,
    make_microduck_backroll_robust_env_cfg,
    make_microduck_headspin_robust_env_cfg,
)


def _delay(cfg):
    a = cfg.scene.entities["robot"].articulation.actuators
    return [(x.delay_min_lag, x.delay_max_lag) for x in a]


def test_tasks_are_registered():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    for t in ("Mjlab-BackRollRobust-Flat-MicroDuck", "Mjlab-HeadSpinRobust-Flat-MicroDuck",
              "Mjlab-BackRoll-Flat-MicroDuck", "Mjlab-HeadSpin-Flat-MicroDuck"):
        assert t in list_tasks(), t


def test_delay_is_widened_and_base_untouched():
    assert all(d == DELAY_RANGE for d in _delay(make_microduck_backroll_robust_env_cfg()))
    assert all(d == DELAY_RANGE for d in _delay(make_microduck_headspin_robust_env_cfg()))
    assert all(d == (3, 6) for d in _delay(make_microduck_backroll_env_cfg()))
    assert all(d == (3, 6) for d in _delay(make_microduck_headspin_env_cfg()))
    assert all(d == DELAY_RANGE for d in _delay(make_microduck_headspin_robust_env_cfg(play=True)))


def test_nothing_else_changes():
    base, rob = make_microduck_backroll_env_cfg(), make_microduck_backroll_robust_env_cfg()
    assert sorted(base.rewards) == sorted(rob.rewards)
    assert {k: v.weight for k, v in base.rewards.items()} == {k: v.weight for k, v in rob.rewards.items()}
    a0 = base.scene.entities["robot"].articulation.actuators[0]
    a1 = rob.scene.entities["robot"].articulation.actuators[0]
    keep = {k: v for k, v in vars(a0).items() if k not in ("delay_min_lag", "delay_max_lag")}
    assert keep == {k: v for k, v in vars(a1).items() if k not in ("delay_min_lag", "delay_max_lag")}
    hs = make_microduck_headspin_robust_env_cfg()
    assert sorted(hs.rewards) == sorted(make_microduck_headspin_env_cfg().rewards)
    assert abs(hs.commands["twist"].min_frac - HEADSPIN_MIN_FRAC) < 1e-9


def test_runners_keep_the_experiment_folders():
    assert MicroduckBackrollRobustRlCfg.experiment_name == MicroduckBackrollRlCfg.experiment_name
    assert MicroduckHeadSpinRobustRlCfg.experiment_name == MicroduckHeadSpinRlCfg.experiment_name
    assert MicroduckBackrollRobustRlCfg.run_name == "microduck_backroll_robust"
    assert MicroduckHeadSpinRobustRlCfg.run_name == "microduck_headspin_robust"
'''

TEXT_11 = r'''"""Fast headspin: only the spin command range moves (1-6 / 1-5 rad/s); the latency-robust
headspin it builds on is untouched. CPU only, no warp env is built."""

import torch

from mjlab_microduck.tasks import headspin_mdp as hsp
from mjlab_microduck.tasks.microduck_headspin_fast_env_cfg import (
    MIN_RAD_S,
    MicroduckHeadSpinFast5RlCfg,
    MicroduckHeadSpinFast6RlCfg,
    make_microduck_headspin_fast5_env_cfg,
    make_microduck_headspin_fast6_env_cfg,
)
from mjlab_microduck.tasks.microduck_robust_env_cfg import (
    DELAY_RANGE,
    HEADSPIN_MIN_FRAC,
    MicroduckHeadSpinRobustRlCfg,
    make_microduck_headspin_robust_env_cfg,
)


def test_tasks_are_registered():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    for t in ("Mjlab-HeadSpinFast6-Flat-MicroDuck", "Mjlab-HeadSpinFast5-Flat-MicroDuck",
              "Mjlab-HeadSpinRobust-Flat-MicroDuck"):
        assert t in list_tasks(), t


def test_command_range():
    for make, top in ((make_microduck_headspin_fast6_env_cfg, 6.0), (make_microduck_headspin_fast5_env_cfg, 5.0)):
        for play in (False, True):
            cfg = make(play=play)
            cmd = cfg.commands["twist"]
            assert tuple(cmd.ranges.ang_vel_z) == (-top, top)
            assert abs(cmd.min_frac * top - MIN_RAD_S) < 1e-9
            stages = cfg.curriculum["headstand_command_ranges"].params["velocity_stages"]
            assert [s["ang_vel_range"] for s in stages] == [top]
            w = hsp.sample_spin_rates(20000, top, cmd.min_frac, "cpu").abs()
            assert float(w.min()) >= MIN_RAD_S - 1e-5 and float(w.max()) <= top + 1e-5


def test_robust_task_untouched_and_rest_equal():
    rob = make_microduck_headspin_robust_env_cfg()
    assert abs(rob.commands["twist"].min_frac - HEADSPIN_MIN_FRAC) < 1e-9
    stages = rob.curriculum["headstand_command_ranges"].params["velocity_stages"]
    assert [s["ang_vel_range"] for s in stages] == [1.0, 2.0, 3.0]
    fast = make_microduck_headspin_fast6_env_cfg()
    assert sorted(fast.rewards) == sorted(rob.rewards)
    assert {k: v.weight for k, v in fast.rewards.items()} == {k: v.weight for k, v in rob.rewards.items()}
    a = fast.scene.entities["robot"].articulation.actuators
    assert all((x.delay_min_lag, x.delay_max_lag) == DELAY_RANGE for x in a)
    assert abs(fast.commands["twist"].zero_frac - rob.commands["twist"].zero_frac) < 1e-9


def test_configs_do_not_share_the_command_range():
    """Building one task must not move another's range (a shared ranges object would)."""
    fast6 = make_microduck_headspin_fast6_env_cfg()
    rob = make_microduck_headspin_robust_env_cfg()
    fast5 = make_microduck_headspin_fast5_env_cfg()
    assert tuple(fast6.commands["twist"].ranges.ang_vel_z) == (-6.0, 6.0)
    assert tuple(rob.commands["twist"].ranges.ang_vel_z) == (-1.0, 1.0)     # stage 0 of the robust task
    assert tuple(fast5.commands["twist"].ranges.ang_vel_z) == (-5.0, 5.0)


def test_runners_keep_the_experiment_folder():
    for rl, name in ((MicroduckHeadSpinFast6RlCfg, "microduck_headspin_fast6"),
                     (MicroduckHeadSpinFast5RlCfg, "microduck_headspin_fast5")):
        assert rl.experiment_name == MicroduckHeadSpinRobustRlCfg.experiment_name
        assert rl.run_name == name


def test_sampling_is_torch():
    assert isinstance(hsp.sample_spin_rates(4, 6.0, 1 / 6, "cpu"), torch.Tensor)
'''

INIT_APPEND = r'''

# [acrobatics] Headstand, backward roll, headspin and their latency-robust / fast versions
# (microduck-acrobatics, added by installers/setup_acrobatics.py).
from .microduck_backroll_env_cfg import (  # noqa: E402
    MicroduckBackrollRlCfg,
    make_microduck_backroll_env_cfg,
)
from .microduck_headspin_env_cfg import (  # noqa: E402
    MicroduckHeadSpinRlCfg,
    make_microduck_headspin_env_cfg,
)
from .microduck_headspin_fast_env_cfg import (  # noqa: E402
    MicroduckHeadSpinFast5RlCfg,
    MicroduckHeadSpinFast6RlCfg,
    make_microduck_headspin_fast5_env_cfg,
    make_microduck_headspin_fast6_env_cfg,
)
from .microduck_headstand_env_cfg import (  # noqa: E402
    MicroduckHeadstandRlCfg,
    make_microduck_headstand_env_cfg,
)
from .microduck_robust_env_cfg import (  # noqa: E402
    MicroduckBackrollRobustRlCfg,
    MicroduckHeadSpinRobustRlCfg,
    make_microduck_backroll_robust_env_cfg,
    make_microduck_headspin_robust_env_cfg,
)

for _task_id, _make, _rl in (
    ("Mjlab-HeadStand-Flat-MicroDuck", make_microduck_headstand_env_cfg, MicroduckHeadstandRlCfg),
    ("Mjlab-BackRoll-Flat-MicroDuck", make_microduck_backroll_env_cfg, MicroduckBackrollRlCfg),
    ("Mjlab-HeadSpin-Flat-MicroDuck", make_microduck_headspin_env_cfg, MicroduckHeadSpinRlCfg),
    ("Mjlab-BackRollRobust-Flat-MicroDuck", make_microduck_backroll_robust_env_cfg, MicroduckBackrollRobustRlCfg),
    ("Mjlab-HeadSpinRobust-Flat-MicroDuck", make_microduck_headspin_robust_env_cfg, MicroduckHeadSpinRobustRlCfg),
    ("Mjlab-HeadSpinFast5-Flat-MicroDuck", make_microduck_headspin_fast5_env_cfg, MicroduckHeadSpinFast5RlCfg),
    ("Mjlab-HeadSpinFast6-Flat-MicroDuck", make_microduck_headspin_fast6_env_cfg, MicroduckHeadSpinFast6RlCfg),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=_make(),
        play_env_cfg=_make(play=True),
        rl_cfg=_rl,
        runner_cls=MicroduckOnPolicyRunner,
    )
'''

# (text that must be in mdp.py, replacement, how many times it must appear)
MDP_EDITS = [
    ('_HEAD_LATCH_LO = math.radians(20.0)\n_HEAD_LATCH_HI = math.radians(170.0)\n', '_HEAD_LATCH_LO = math.radians(20.0)\n_HEAD_LATCH_HI = math.radians(170.0)\n\n\n# [backroll] Roll direction and head-latch window are per ENV, written by\n# reset_roulade_state (defaults = the forward roulade, unchanged). A backward\n# roll plants the head late in the roll, so its latch window moves.\ndef _roulade_sign(env) -> float:\n    return _ROULADE_FWD_SIGN * getattr(env, "_roulade_direction", 1.0)\n\n\ndef _roulade_latch_window(env) -> tuple:\n    return getattr(env, "_roulade_latch_window", (_HEAD_LATCH_LO, _HEAD_LATCH_HI))\n', 1),
    ('_ROULADE_FWD_SIGN * asset.data.root_link_ang_vel_b[:, 1]', '_roulade_sign(env) * asset.data.root_link_ang_vel_b[:, 1]', 2),
    ('            in_window = (env._roulade_accum > _HEAD_LATCH_LO) & (\n                env._roulade_accum < _HEAD_LATCH_HI\n            )\n', '            latch_lo, latch_hi = _roulade_latch_window(env)  # [backroll]\n            in_window = (env._roulade_accum > latch_lo) & (env._roulade_accum < latch_hi)\n', 1),
    ('    tuck_factor_range: tuple = (0.3, 1.0),\n    joint_noise_std: float = 0.0,\n):\n    """Reset to a standing start or a mid-roll state (reverse curriculum).\n', '    tuck_factor_range: tuple = (0.3, 1.0),\n    joint_noise_std: float = 0.0,\n    direction: float = 1.0,\n    head_latch_lo: float = _HEAD_LATCH_LO,\n    head_latch_hi: float = _HEAD_LATCH_HI,\n):\n    """Reset to a standing start or a mid-roll state (reverse curriculum).\n\n    [backroll] ``direction``: +1 = forward roll (the roulade, default), -1 =\n    backward roll. Mid-roll spawns are pitched that way with angular momentum\n    that way, and the accumulator counts rotation in that direction.\n    ``head_latch_lo/hi``: progress window in which head-top contact latches.\n', 1),
    ('    asset: Entity = env.scene[asset_cfg.name]\n    accum, max_accum, paid = _roulade_state(env)\n', '    asset: Entity = env.scene[asset_cfg.name]\n    accum, max_accum, paid = _roulade_state(env)\n    env._roulade_direction = float(direction)  # [backroll]\n    env._roulade_latch_window = (float(head_latch_lo), float(head_latch_hi))\n', 1),
    ('    pitch = torch.where(is_mid, mid_pitch, pitch)\n', '    pitch = torch.where(is_mid, direction * mid_pitch, pitch)  # [backroll] signed\n', 1),
    ('        env.sim.data.qvel[mid_env_ids, 4] = _ROULADE_FWD_SIGN * omega\n', '        env.sim.data.qvel[mid_env_ids, 4] = _ROULADE_FWD_SIGN * direction * omega  # [backroll]\n', 1),
    ('        env.sim.data.qvel[stand_env_ids, 0] = vx * torch.cos(yaw_s)\n        env.sim.data.qvel[stand_env_ids, 1] = vx * torch.sin(yaw_s)\n', '        env.sim.data.qvel[stand_env_ids, 0] = direction * vx * torch.cos(yaw_s)  # [backroll]\n        env.sim.data.qvel[stand_env_ids, 1] = direction * vx * torch.sin(yaw_s)\n', 1),
    ('    env._roulade_head_latch[env_ids] = is_mid\n', '    env._roulade_head_latch[env_ids] = is_mid & (mid_pitch >= head_latch_lo)  # [backroll]\n', 1),
]


FILES = [
    ("src/mjlab_microduck/tasks/headstand_mdp.py", TEXT_00),
    ("src/mjlab_microduck/tasks/microduck_headstand_env_cfg.py", TEXT_01),
    ("src/mjlab_microduck/tasks/microduck_backroll_env_cfg.py", TEXT_02),
    ("src/mjlab_microduck/tasks/headspin_mdp.py", TEXT_03),
    ("src/mjlab_microduck/tasks/microduck_headspin_env_cfg.py", TEXT_04),
    ("src/mjlab_microduck/tasks/microduck_robust_env_cfg.py", TEXT_05),
    ("src/mjlab_microduck/tasks/microduck_headspin_fast_env_cfg.py", TEXT_06),
    ("tests/test_headstand_cfg.py", TEXT_07),
    ("tests/test_backroll_cfg.py", TEXT_08),
    ("tests/test_headspin_cfg.py", TEXT_09),
    ("tests/test_robust_cfg.py", TEXT_10),
    ("tests/test_headspin_fast_cfg.py", TEXT_11),
]


def fail(*lines):
    print("NG: " + lines[0])
    for line in lines[1:]:
        print("    " + line)
    print("Nothing was changed.")
    sys.exit(1)


def payload_digest():
    h = hashlib.sha256()
    for rel, text in FILES:
        h.update(rel.encode("utf-8"))
        h.update(text.encode("utf-8"))
    h.update(INIT_APPEND.encode("utf-8"))
    for old, rep, count in MDP_EDITS:
        h.update(old.encode("utf-8"))
        h.update(rep.encode("utf-8"))
        h.update(str(count).encode("utf-8"))
    return h.hexdigest()


MDP_REL = "src/mjlab_microduck/tasks/mdp.py"
MDP_MARK = "[backroll] Roll direction"       # present once the edits below are in


# Things the new files rely on in the microduck_rl clone.
NEEDED = [
    ("src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py",
     ("def make_microduck_roulade_env_cfg(", "TUCK_OVERRIDES", "MicroduckRouladeRlCfg =")),
    ("src/mjlab_microduck/tasks/mdp.py",
     ("def reset_roulade_state(", "class VelocityCommandCommandOnly", "def velocity_command_ranges_curriculum(",
      "def event_param_curriculum(", "def reward_weight(")),
    ("src/mjlab_microduck/robot/microduck_constants.py", ("delay_min_lag=3", "delay_max_lag=6")),
    ("src/mjlab_microduck/tasks/__init__.py", ("register_mjlab_task", "MicroduckOnPolicyRunner")),
]
# Task registrations of earlier per-task installers: they would collide with ours.
OLD_TAGS = ("[headstand]", "[backroll]", "[headspin]", "[robust]", "[headspin_fast]")


def main():
    root = Path.cwd()
    if payload_digest() != PAYLOAD_SHA256:
        fail("this script was not copied exactly (a line is missing or changed).")
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file() or "mjlab-microduck" not in pyproject.read_text(encoding="utf-8"):
        fail("run this inside the microduck_rl folder:  cd microduck_rl")
    for rel, needles in NEEDED:
        p = root / rel
        if not p.is_file():
            fail(f"{rel} not found")
        text = p.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                fail(f"{rel} does not look like the tested version (missing: {needle})")
    init = root / "src" / "mjlab_microduck" / "tasks" / "__init__.py"
    init_text = init.read_text(encoding="utf-8")
    old = [t for t in OLD_TAGS if t in init_text]
    if old:
        fail("tasks were already registered by older installers: " + ", ".join(old),
             "Use a fresh clone of microduck_rl.")

    # The backward roll needs the roulade's reset / latch helpers in mdp.py to take a
    # roll direction and a head-latch window (defaults = the forward roulade, unchanged).
    mdp_path = root / MDP_REL
    mdp = mdp_path.read_text(encoding="utf-8")
    new_mdp = None
    if MDP_MARK not in mdp:
        new_mdp = mdp
        for old, rep, count in MDP_EDITS:
            found = new_mdp.count(old)
            if found != count:
                fail(f"mdp.py: expected {count} match(es), found {found}, for:", old.strip().splitlines()[0])
            new_mdp = new_mdp.replace(old, rep)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True).stdout.strip()
    if head != PINNED:
        print(f"WARN: repository is at {head[:7] or '?'}; this script was tested on {PINNED[:7]}.")

    changed = 0
    if new_mdp is None:
        print(f"[OK] {MDP_REL} (roll direction already there)")
    else:
        mdp_path.write_text(new_mdp, encoding="utf-8")
        changed += 1
        print(f"[OK] {MDP_REL}: roll direction and head-latch window are now per-task settings")
    for rel, text in FILES:
        p = root / rel
        if p.is_file() and p.read_text(encoding="utf-8") == text:
            print(f"[OK] {rel} (already there)")
            continue
        p.write_text(text, encoding="utf-8")
        changed += 1
        print(f"[OK] wrote {rel}")
    if TAG in init_text:
        print("[OK] tasks already registered")
    else:
        with init.open("a", encoding="utf-8") as f:
            f.write(INIT_APPEND)
        changed += 1
        print("[OK] registered 7 tasks")
    if changed == 0:
        print("Already installed. Nothing to do.")
    print("DONE")


if __name__ == "__main__":
    main()
