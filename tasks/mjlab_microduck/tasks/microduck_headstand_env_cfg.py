"""Microduck headstand: stand on the flat top of the head and hold the balance.

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
