"""Microduck headspin: spin on the head top at a commanded rate, stop on command.

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
