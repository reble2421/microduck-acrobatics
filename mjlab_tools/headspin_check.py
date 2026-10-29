"""Headspin env on CPU: command buckets, then score real policies with the headspin terms.

    cd microduck_rl   (a clone with installers/setup_acrobatics.py applied)
    PYTHONPATH=$PWD/src python ../microduck-acrobatics/mjlab_tools/headspin_check.py ONNX [ONNX ...]

Per policy (4 envs, headstand spawns): 1 s zero, 5 s spin +2 rad/s, 5 s spin -2 rad/s,
4 s zero. Prints, per segment, the achieved yaw rate, COM drift and the mean
headspin_track / spin_when_stopped values.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mjlab_lib import OnnxMLP, build_env  # noqa: E402

from mjlab_microduck.tasks import headspin_mdp as hsp  # noqa: E402

TASK = "Mjlab-HeadSpin-Flat-MicroDuck"
N = 4
# HS_DELAY="lo,hi" fixes the actuator command delay (physics steps), e.g. "0,0"; default: the task's 3-6
DELAY = tuple(int(x) for x in os.environ["HS_DELAY"].split(",")) if os.environ.get("HS_DELAY") else None
env, term = build_env(num_envs=N, task=TASK, delay=DELAY)
_act = env.cfg.scene.entities["robot"].articulation.actuators[0]
print(f"command delay: {_act.delay_min_lag}-{_act.delay_max_lag} physics steps")
dt = float(env.step_dt)

# ---- command buckets (the class's own sampling) ------------------------------------
ids = torch.arange(N)
zero = total = 0
mags = []
for _ in range(500):
    term._resample_command(ids)
    z = term.is_standing_env
    zero += int(z.sum())
    total += N
    mags.append(term.vel_command_b[~z, 2].abs())
    assert float(term.vel_command_b[:, :2].abs().max()) == 0.0
m = torch.cat(mags)
print(f"buckets: zero {100 * zero / total:.0f}% (designed 25%), spin |wz| {float(m.min()):.2f}-{float(m.max()):.2f} rad/s "
      f"(stage-0 range {term.cfg.ranges.ang_vel_z[1]}, min_frac {term.cfg.min_frac})")

SEGMENTS = [("zero", 0.0, 1.0),
            ("spin +1", 1.0, 4.0), ("spin -1", -1.0, 4.0),
            ("spin +2", 2.0, 4.0), ("spin -2", -2.0, 4.0),
            ("spin +3", 3.0, 4.0), ("spin -3", -3.0, 4.0),
            ("stop", 0.0, 4.0)]
for path in sys.argv[1:]:
    policy = OnnxMLP(path)
    obs, _ = env.reset()
    print(f"\n{os.path.basename(path)}")
    print(f"  {'segment':8s} {'yaw deg/s':>10s} {'ratio':>6s} {'COM drift mm/s':>15s} {'track':>7s} {'spin cost':>10s} {'on head':>8s}")
    for name, wz, secs in SEGMENTS:
        acc = {"yaw": 0.0, "drift": 0.0, "track": 0.0, "cost": 0.0, "on": 0.0}
        n = 0
        for k in range(int(round(secs / dt))):
            term.vel_command_b[:] = torch.tensor([0.0, 0.0, wz])
            term.is_standing_env[:] = wz == 0.0
            with torch.no_grad():
                obs, *_ = env.step(policy(obs["actor"]))
            if k * dt < 1.0 and secs > 1.0:
                continue
            acc["yaw"] += float(env._hs_ema[:, 2].mean())
            acc["drift"] += float(torch.linalg.norm(env._hsp_com_v, dim=-1).mean()) if hasattr(env, "_hsp_com_v") else 0.0
            acc["track"] += float(hsp.headspin_track(env).mean())
            acc["cost"] += float(hsp.spin_when_stopped(env).mean())
            acc["on"] += float(env._hs_on.float().mean())
            n += 1
        a = {k: v / max(n, 1) for k, v in acc.items()}
        ratio = f"{a['yaw'] / wz:+6.2f}" if wz else "     -"
        print(f"  {name:8s} {math.degrees(a['yaw']):+10.0f} {ratio} {1000 * a['drift']:15.1f} {a['track']:+7.3f} "
              f"{a['cost']:10.3f} {a['on']:8.2f}", flush=True)
