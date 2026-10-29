"""Latency-robust backroll / headspin: the delay range is widened, nothing else moves,
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
