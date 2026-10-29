"""Fast headspin: the latency-robust headspin with spin commands up to 5 or 6 rad/s.

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
