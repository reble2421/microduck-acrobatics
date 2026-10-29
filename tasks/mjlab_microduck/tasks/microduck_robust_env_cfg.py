"""Latency-robust versions of the backroll and the headspin.

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
