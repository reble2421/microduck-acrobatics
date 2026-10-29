#!/usr/bin/env python3
"""Checks that the bench simulation matches microduck_rl's own rehearsal.

    python bench/tests/test_sim.py      (MICRODUCK_RL and MICRODUCK_POLICIES set, see README)

1. The 61-D observation equals PolicyInference.get_observations() on the same
   state, after steps that move every joint (so every slot is non-trivial).
2. A standing start under the published stand policy stays up.
3. The roll judge counts a scripted rotation correctly.
"""

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from battery import PUBLISHED  # noqa: E402
from duckbench import sim as S  # noqa: E402
from duckbench.scenarios import make_roll_judge  # noqa: E402

STAND = str(PUBLISHED / "alpha_stand.onnx")


def test_observation_matches_policy_inference():
    duck = S.DuckSim()
    duck.reset("stand", yaw=0.7, joint_noise=0.05, tilt=math.radians(3), rng=np.random.default_rng(1))
    policy = S.Policy(STAND)
    with S.quiet():
        ref = S.ip.PolicyInference(duck.model, duck.data, walking_onnx_path=STAND, bam_ctrl=duck.bam,
                                   new_cmd_obs=True, use_projected_gravity=True)
    cmd = np.zeros(13, dtype=np.float32)
    cmd[3:7] = (0.2, -0.1, 0.3, 0.05)   # head pose slots
    cmd[7:13] = (0.0, 0.0, -0.01, 0.1, 0.05, 0.0)  # body pose slots
    for k in range(30):
        mine = duck.observation(cmd)
        ref.last_action = duck.last_action.copy()
        ref.command = cmd.copy()
        theirs = ref.get_observations()
        assert mine.shape == (61,), mine.shape
        err = float(np.abs(mine - theirs).max())
        assert err < 1e-6, f"step {k}: max |obs diff| = {err}"
        duck.step(policy, cmd)
    print("OK  observation == PolicyInference.get_observations() over 30 steps")


def test_stand_policy_stays_up():
    duck = S.DuckSim()
    duck.reset("stand")
    policy = S.Policy(STAND)
    for _ in range(int(3.0 / S.CONTROL_DT)):
        duck.step(policy, np.zeros(13, dtype=np.float32))
    st = duck.state()
    assert st["z"] > 0.10 and st["upright"] > 0.95, st
    print(f"OK  stand policy: z={st['z']:.3f} upright={st['upright']:.3f} after 3 s")


def test_roll_judge_counts_rotation():
    n = 250
    t = np.arange(1, n + 1) * S.CONTROL_DT
    rate = np.zeros(n)
    rate[:100] = -2 * math.pi / (100 * S.CONTROL_DT)     # one backward turn in 2 s
    tr = {"t": t, "pitch_rate": rate, "z": np.full(n, 0.12), "upright": np.ones(n),
          "trunk_yaw": np.zeros(n)}
    back = make_roll_judge(-1)(tr)
    fwd = make_roll_judge(+1)(tr)
    assert back["success"] and abs(back["roll_deg"] - 360) < 1, back
    assert not fwd["success"] and abs(fwd["wrong_way_deg"] - 360) < 1, fwd
    print(f"OK  roll judge: backward {back['roll_deg']} deg counted, forward judge rejects it")


if __name__ == "__main__":
    test_observation_matches_policy_inference()
    test_stand_policy_stays_up()
    test_roll_judge_counts_rotation()
    print("all passed")
