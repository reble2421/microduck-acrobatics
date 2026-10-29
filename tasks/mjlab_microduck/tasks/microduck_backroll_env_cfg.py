"""Microduck backward roll ("back roll") - the roulade, run the other way.

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
