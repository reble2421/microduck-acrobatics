"""Fast headspin: only the spin command range moves (1-6 / 1-5 rad/s); the latency-robust
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
