"""A routine: tricks chained by hot-swapping ONNX policies, with a check after each stage.

Every stage runs one policy for a fixed time and may end with a check on the trace
so far. A failed check ends the trial there, so the per-stage numbers form a
funnel: how often each transition succeeds GIVEN that everything before it did.
The checks reuse the thresholds of scenarios.py and are independent of the
training rewards.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .scenarios import (FALLEN_UPRIGHT, FALLEN_Z, STAND_Z_MIN, UPRIGHT_MIN, ZERO_CMD,
                        _first_run, twist)
from .sim import CONTROL_DT


@dataclass
class Stage:
    name: str
    label: str                     # shown in videos (Japanese)
    role: str                      # policy role; "hold" = keep the default pose
    seconds: float
    command: np.ndarray = None
    check: Optional[Callable] = None   # check(tr, start) -> (ok, info); start[name] = first index

    def cmd(self):
        return ZERO_CMD if self.command is None else self.command


def _standing(tr, idx):
    return (tr["z"][idx] >= STAND_Z_MIN) & (tr["upright"][idx] >= UPRIGHT_MIN)


# ---- checks -----------------------------------------------------------------------

def check_walked(stage, min_travel=0.05):
    def check(tr, start):
        sl = slice(start[stage], None)
        fallen = (tr["z"][sl] < FALLEN_Z) | (tr["upright"][sl] < FALLEN_UPRIGHT)
        travel = math.hypot(tr["x"][-1] - tr["x"][start[stage]], tr["y"][-1] - tr["y"][start[stage]])
        return (not fallen.any()) and travel >= min_travel, {"travel_m": round(travel, 3)}
    return check


def check_standing(hold_s=0.5):
    def check(tr, start):
        n = max(1, int(round(hold_s / CONTROL_DT)))
        idx = np.arange(len(tr["z"]))[-n:]
        return bool(_standing(tr, idx).all()), {"z": round(float(tr["z"][-1]), 3)}
    return check


def check_rolled(from_stage, direction, min_deg=300.0, hold_s=0.5):
    """A full roll since from_stage started, then standing for hold_s."""
    def check(tr, start):
        rate = tr["pitch_rate"][start[from_stage]:]
        accum = np.cumsum(direction * rate) * CONTROL_DT
        deg = math.degrees(float(accum.max()))
        n = max(1, int(round(hold_s / CONTROL_DT)))
        standing = bool(_standing(tr, np.arange(len(tr["z"]))[-n:]).all())
        return deg >= min_deg and standing, {"roll_deg": round(deg, 1), "standing": standing}
    return check


def check_headstand(stage, entry_deadline_s=3.0, min_fraction=0.9, hold_s=0.5):
    def check(tr, start):
        on = tr["on_head"][start[stage]:].astype(bool)
        n = max(1, int(round(hold_s / CONTROL_DT)))
        entry = _first_run(on, n)
        if entry is None:
            return False, {"entry_s": None, "on_head_fraction": 0.0}
        frac = float(on[entry:].mean())
        ok = entry * CONTROL_DT <= entry_deadline_s and frac >= min_fraction and bool(on[-n:].all())
        return ok, {"entry_s": round(entry * CONTROL_DT, 2), "on_head_fraction": round(frac, 3)}
    return check


def check_off_head():
    def check(tr, start):
        return not bool(tr["on_head"][-1]), {}
    return check


def _yaw_rate(tr, a, b):
    """Mean head-heading rate (rad/s) between indices a and b (world yaw, + = counter-clockwise from above)."""
    d = np.diff(tr["heading"][a:b + 1])
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float(d.sum()) / max((b - a) * CONTROL_DT, CONTROL_DT)


def check_spun(stage, wz, min_ratio=0.7, min_fraction=0.9, skip_s=1.0):
    """Spinning on the head at >= min_ratio of the commanded rate (after skip_s), still on the head."""
    def check(tr, start):
        a = start[stage] + int(round(skip_s / CONTROL_DT))
        b = len(tr["heading"]) - 1
        on = tr["on_head"][start[stage]:].astype(bool)
        rate = _yaw_rate(tr, a, b) if b > a else 0.0
        frac = float(on.mean())
        ok = rate / wz >= min_ratio and frac >= min_fraction and bool(on[-1])
        return ok, {"ratio": round(rate / wz, 2), "deg_s": round(math.degrees(rate), 1),
                    "on_head_fraction": round(frac, 3)}
    return check


def check_spin_stopped(stage, max_deg_s=15.0, last_s=1.0):
    """Stopped spinning (residual yaw rate over the last last_s) and still on the head."""
    def check(tr, start):
        b = len(tr["heading"]) - 1
        a = max(start[stage], b - int(round(last_s / CONTROL_DT)))
        rate = math.degrees(_yaw_rate(tr, a, b))
        on = bool(tr["on_head"][a:].astype(bool).all())
        return abs(rate) <= max_deg_s and on, {"deg_s": round(rate, 1), "on_head": on}
    return check


# ---- the routine --------------------------------------------------------------------

def build_routine(roll_role="backroll", roll_direction=-1.0, roll_s=2.0, release_s=1.0,
                  headstand_s=6.0, headstand_cmd=None, walk_vx=0.3, recover_s=2.0,
                  spins=(), spin_stop_s=2.0):
    """spins: ((wz rad/s, seconds), ...) spun on the head after the headstand, each checked,
    then a stop (zero command) for spin_stop_s; the "headstand" role must be a headspin policy."""
    roll_label = "後転" if roll_direction < 0 else "前転"
    stages = [
        Stage("walk", "歩く", "walk", 3.0, twist(vx=walk_vx), check_walked("walk")),
        Stage("stop", "止まる", "stand", 1.5, None, check_standing()),
        Stage("roll", roll_label, roll_role, roll_s),
        Stage("recover", f"{roll_label}から立つ", "stand", recover_s, None,
              check_rolled("roll", roll_direction)),
        Stage("headstand", "逆立ち", "headstand", headstand_s, headstand_cmd, check_headstand("headstand")),
    ]
    for i, (wz, secs) in enumerate(spins):
        side = "左回り" if wz > 0 else "右回り"     # + = counter-clockwise seen from above
        name = f"spin{i + 1}"
        stages.append(Stage(name, f"ヘッドスピン（{side}）", "headstand", secs, twist(wz=wz),
                            check_spun(name, wz)))
    if spins:
        stages.append(Stage("spin_stop", "回転を止める", "headstand", spin_stop_s, None,
                            check_spin_stopped("spin_stop")))
    if release_s > 0:
        stages.append(Stage("release", "倒れる", "hold", release_s, None, check_off_head()))
    stages.append(Stage("standup", "起き上がる", "stand", 5.0, None, check_standing(hold_s=1.0)))
    return stages
