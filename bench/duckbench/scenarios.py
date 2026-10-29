"""What a trick is: a start state, a policy schedule, and a fixed success judge.

The judges are written by hand and are deliberately independent of the
training rewards, so a policy that farms a reward term still has to pass them.
Every judge returns a dict with at least {"success": bool}; the other keys are
the numbers behind the verdict, written to trials.csv.
"""

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .sim import CONTROL_DT

ZERO_CMD = np.zeros(13, dtype=np.float32)

STAND_Z_MIN = 0.10     # trunk origin height of a stand (0.125 at spawn)
UPRIGHT_MIN = 0.90     # cos(tilt) of a stand: within ~25 deg of vertical
FALLEN_Z = 0.07        # headless_rollout.py's "fell over" verdict
FALLEN_UPRIGHT = 0.5


def twist(vx=0.0, vy=0.0, wz=0.0):
    cmd = ZERO_CMD.copy()
    cmd[0:3] = (vx, vy, wz)
    return cmd


@dataclass
class Segment:
    role: str                      # which policy runs (key of the policy map)
    seconds: float
    command: object = None         # 13-D array, callable(sim, ctx) -> 13-D array, or None (= zero)
    label: str = ""
    gate_on_head: bool = False     # send the command only while on the head

    def command_at(self, sim, ctx):
        if self.gate_on_head and not ctx.get("on_head", False):
            return ZERO_CMD
        if self.command is None:
            return ZERO_CMD
        if callable(self.command):
            return self.command(sim, ctx)
        return self.command


@dataclass
class Scenario:
    name: str
    start: str                     # DuckSim.reset start state
    segments: list
    judge: Callable
    description: str = ""
    camera: dict = field(default_factory=dict)

    @property
    def roles(self):
        return sorted({s.role for s in self.segments})

    @property
    def seconds(self):
        return sum(s.seconds for s in self.segments)


# ---- helpers --------------------------------------------------------------------

def _first_run(mask, n):
    """Index where the first run of n consecutive True values starts, or None."""
    if n <= 1:
        idx = np.flatnonzero(mask)
        return int(idx[0]) if len(idx) else None
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= n:
            return i - n + 1
    return None


def _standing(tr):
    return (tr["z"] >= STAND_Z_MIN) & (tr["upright"] >= UPRIGHT_MIN)


def _time(tr, idx):
    return None if idx is None else round(float(tr["t"][idx]), 3)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---- judges ------------------------------------------------------------------------

def judge_stays_up(tr):
    fallen = (tr["z"] < FALLEN_Z) | (tr["upright"] < FALLEN_UPRIGHT)
    fell = _first_run(fallen, 1)
    return {
        "success": fell is None,
        "fell_at_s": _time(tr, fell),
        "final_z": round(float(tr["z"][-1]), 4),
        "final_upright": round(float(tr["upright"][-1]), 3),
        "travel_x_m": round(float(tr["x"][-1] - tr["x"][0]), 3),
    }


def make_roll_judge(direction, min_deg=300.0, settle_s=0.5):
    """A full roll in `direction` (+1 forward, -1 backward), then standing.

    Rotation is integrated from the trunk's body-frame pitch rate, the same
    quantity the roulade reward accumulates (mdp._roulade_state). Success:
    at least min_deg of rotation in the right direction, and standing
    (height and uprightness) for the last settle_s of the trial.
    """
    def judge(tr):
        accum = np.cumsum(direction * tr["pitch_rate"]) * CONTROL_DT
        done = _first_run(accum >= math.radians(min_deg), 1)
        n_tail = max(1, int(round(settle_s / CONTROL_DT)))
        end_standing = bool(_standing(tr)[-n_tail:].all())
        # How clean was it? A real roll lands on the feet: little time lying on the
        # back/side after going over, and the heading barely changes. "Flop, then get
        # up some other way" lies for 0.4-1.2 s and turns ~150 deg (roulade cut at 1 s).
        up, z = tr["upright"], tr["z"]
        inv = np.flatnonzero(up < -0.5)
        lying, t_land = 0.0, None
        if len(inv):
            i0 = int(inv[0])
            lying = float(((np.abs(up[i0:]) < 0.5) & (z[i0:] < 0.08)).sum()) * CONTROL_DT
            st = np.flatnonzero(_standing(tr)[i0:])
            t_land = round(float(st[0]) * CONTROL_DT, 2) if len(st) else None
        yaw = math.degrees(float(_wrap(tr["trunk_yaw"][-1] - tr["trunk_yaw"][0])))
        net = math.degrees(float(accum[-1]))
        clean = bool(net >= min_deg and lying <= 0.4 and abs(yaw) <= 90 and end_standing)
        return {
            "success": done is not None and end_standing,
            "clean": clean,
            "roll_deg": round(math.degrees(float(accum.max())), 1),
            "net_deg": round(net, 1),
            "wrong_way_deg": round(math.degrees(float(max(0.0, -accum.min()))), 1),
            "roll_done_s": _time(tr, done),
            "lying_s": round(lying, 2),
            "land_s": t_land,
            "yaw_change_deg": round(yaw, 1),
            "end_standing": end_standing,
            "final_z": round(float(z[-1]), 4),
            "final_upright": round(float(up[-1]), 3),
        }
    return judge


def make_headstand_judge(entry_deadline_s=6.0, min_fraction=0.9, hold_s=0.5):
    """Up on the head top (ONLY the head touching the floor) and staying there.

    entry: the first moment the robot is on its head for hold_s in a row.
    Success: entered by entry_deadline_s, on the head for at least
    min_fraction of the time after entry, and still on the head at the end.
    """
    def judge(tr):
        on = tr["on_head"].astype(bool)
        n_hold = max(1, int(round(hold_s / CONTROL_DT)))
        entry = _first_run(on, n_hold)
        out = {
            "success": False,
            "entry_s": _time(tr, entry),
            "on_head_fraction": 0.0,
            "on_head_at_end": bool(on[-n_hold:].all()),
            "head_drift_m": None,
        }
        if entry is None:
            return out
        frac = float(on[entry:].mean())
        drift = math.hypot(tr["head_x"][-1] - tr["head_x"][entry], tr["head_y"][-1] - tr["head_y"][entry])
        out.update(on_head_fraction=round(frac, 3), head_drift_m=round(drift, 3))
        out["success"] = bool(tr["t"][entry] <= entry_deadline_s and frac >= min_fraction
                              and out["on_head_at_end"])
        return out
    return judge


def make_spin_judge(segments, min_ratio=0.7, max_stop_deg_s=15.0, min_on_head=0.9, skip_s=1.0):
    """Headspin on command. Segment 0 is the entry / settle (zero command). For every
    later segment, after the first skip_s: net heading change / time vs the commanded
    yaw rate (<label>_ratio), or, for a zero command, the residual yaw rate
    (<label>_deg_s, must stay below max_stop_deg_s), plus the whole-robot COM drift
    (<label>_drift_mm_s) and the time on the head (<label>_on_head)."""
    def judge(tr):
        seg, on = tr["segment"], tr["on_head"].astype(bool)
        n_skip = int(round(skip_s / CONTROL_DT))
        out = {"success": False, "entered": bool(on[seg == 0][-10:].all())}
        ok = out["entered"]
        for i, s in enumerate(segments):
            if i == 0:
                continue
            wz = float(np.asarray(s.command)[2]) if s.command is not None else 0.0
            idx = np.flatnonzero(seg == i)[n_skip:]
            frac = float(on[idx].mean()) if len(idx) else 0.0
            out[f"{s.label}_on_head"] = round(frac, 2)
            if len(idx) < 25:
                ok = False
                continue
            a, b = int(idx[0]), int(idx[-1])
            span = (b - a) * CONTROL_DT
            rate = float(np.sum(_wrap(np.diff(tr["heading"][a:b + 1])))) / span
            drift = math.hypot(tr["com_x"][b] - tr["com_x"][a], tr["com_y"][b] - tr["com_y"][a]) / span
            out[f"{s.label}_drift_mm_s"] = round(1000 * drift, 1)
            if wz != 0.0:
                ratio = rate / wz
                out[f"{s.label}_ratio"] = round(ratio, 2)
                good = ratio >= min_ratio
            else:
                out[f"{s.label}_deg_s"] = round(math.degrees(rate), 1)
                good = abs(math.degrees(rate)) <= max_stop_deg_s
            ok = ok and good and frac >= min_on_head
        out["success"] = bool(ok)
        return out
    return judge


# ---- the scenarios --------------------------------------------------------------------

def build_scenarios(handoff_s=1.0):
    """handoff_s: how long an episodic trick runs before the stand policy takes
    over in the *_handoff scenarios (the runtime's manifest duration_s; the
    published roulade uses 1.0 s).

    Headspin segment labels: "right" = +wz and "left" = -wz (names kept from the
    first runs, so they match the saved result files; +wz is counter-clockwise seen
    from above). "pos" / "neg" in the fast scenario mean the same signs."""
    side = {"azimuth": 90.0, "distance": 0.8, "elevation": -8.0}   # rolls read best side-on
    spin = [
        Segment("headstand", 2.0, label="settle"),
        Segment("headstand", 5.0, twist(wz=2.0), "right2", gate_on_head=True),
        Segment("headstand", 5.0, twist(wz=-2.0), "left2", gate_on_head=True),
        Segment("headstand", 5.0, twist(wz=3.0), "right3", gate_on_head=True),
        Segment("headstand", 5.0, twist(wz=-3.0), "left3", gate_on_head=True),
        Segment("headstand", 4.0, None, "stop"),
    ]
    spin_from_stand = [Segment("headstand", 4.0, label="entry")] + spin[1:]
    spin_slow = [
        Segment("headstand", 2.0, label="settle"),
        Segment("headstand", 6.0, twist(wz=1.0), "right1", gate_on_head=True),
        Segment("headstand", 6.0, twist(wz=-1.0), "left1", gate_on_head=True),
        Segment("headstand", 4.0, None, "stop"),
    ]
    # faster than training (|wz| <= 3): pos = + wz (counter-clockwise from above), neg = - wz
    spin_fast = [Segment("headstand", 2.0, label="settle")]
    for w in (4.0, 5.0, 6.0):
        spin_fast += [Segment("headstand", 4.0, twist(wz=w), f"pos{w:.0f}", gate_on_head=True),
                      Segment("headstand", 4.0, twist(wz=-w), f"neg{w:.0f}", gate_on_head=True)]
    spin_fast.append(Segment("headstand", 4.0, None, "stop"))
    return {s.name: s for s in [
        Scenario("headspin", "headstand", spin, make_spin_judge(spin),
                 "headspin on command from a balanced headstand: +/-2 and +/-3 rad/s, 5 s each, then stop",
                 {"azimuth": 135.0, "distance": 0.8, "elevation": -30.0}),
        Scenario("headspin_slow", "headstand", spin_slow, make_spin_judge(spin_slow),
                 "slow headspin: +/-1 rad/s, 6 s each, then stop",
                 {"azimuth": 135.0, "distance": 0.8, "elevation": -30.0}),
        Scenario("headspin_fast", "headstand", spin_fast, make_spin_judge(spin_fast),
                 "fast headspin beyond training: +/-4, +/-5, +/-6 rad/s, 4 s each, then stop",
                 {"azimuth": 135.0, "distance": 0.8, "elevation": -30.0}),
        Scenario("headspin_from_stand", "stand", spin_from_stand, make_spin_judge(spin_from_stand),
                 "as headspin, but starting from standing (entry first)",
                 {"azimuth": 135.0, "distance": 0.8, "elevation": -30.0}),
        Scenario("stand", "stand", [Segment("stand", 6.0)], judge_stays_up,
                 "hold a stand (published alpha_stand)"),
        Scenario("walk", "stand", [Segment("walk", 8.0, twist(vx=0.3))], judge_stays_up,
                 "walk forward at the 0.3 m/s command (published alpha_walking)"),
        Scenario("roulade", "stand", [Segment("roulade", 5.0)], make_roll_judge(+1),
                 "forward roll, the policy alone for its 5 s training episode", side),
        Scenario("roulade_handoff", "stand",
                 [Segment("roulade", handoff_s), Segment("stand", 3.0)], make_roll_judge(+1),
                 "forward roll for handoff_s, then the stand policy takes over (runtime)", side),
        Scenario("backroll", "stand", [Segment("backroll", 5.0)], make_roll_judge(-1),
                 "backward roll, the policy alone for its 5 s training episode", side),
        Scenario("backroll_handoff", "stand",
                 [Segment("backroll", handoff_s), Segment("stand", 3.0)], make_roll_judge(-1),
                 "backward roll for handoff_s, then the stand policy takes over (runtime)", side),
        Scenario("headstand", "stand", [Segment("headstand", 12.0)], make_headstand_judge(),
                 "from standing, get up on the head top and hold still (zero command)", side),
        Scenario("headstand_hold", "headstand", [Segment("headstand", 12.0)],
                 make_headstand_judge(entry_deadline_s=1.0),
                 "start balanced on the head, hold still (zero command)", side),
        Scenario("headstand_long", "stand", [Segment("headstand", 30.0)], make_headstand_judge(),
                 "like headstand, but 30 s: beyond the 12 s training episode", side),
        Scenario("standup_face_down", "face_down", [Segment("standup", 6.0)], make_standup_judge(),
                 "stand up from lying face down (dropped from 0.22 m, as in training)"),
        Scenario("standup_face_up", "face_up", [Segment("standup", 6.0)], make_standup_judge(),
                 "stand up from lying face up (dropped from 0.22 m, as in training)"),
        Scenario("standup_floor_down", "face_down", [Segment("hold", 1.5), Segment("standup", 6.0)],
                 make_standup_judge(), "lie face down on the floor for 1.5 s, then stand up"),
        Scenario("standup_floor_up", "face_up", [Segment("hold", 1.5), Segment("standup", 6.0)],
                 make_standup_judge(), "lie face up on the floor for 1.5 s, then stand up"),
    ]}


def make_standup_judge(hold_s=1.0):
    """From lying on the floor to a stand that is held for hold_s."""
    def judge(tr):
        n_hold = max(1, int(round(hold_s / CONTROL_DT)))
        stand = _standing(tr)
        up = _first_run(stand, n_hold)
        last_pre = np.flatnonzero(tr["segment"] == 0)[-1]      # end of a settle segment, if any
        return {
            "success": bool(stand[-n_hold:].all()),
            "upright_before": round(float(tr["upright"][last_pre]), 2) if tr["segment"][-1] > 0 else None,
            "stood_up_s": _time(tr, up),
            "final_z": round(float(tr["z"][-1]), 4),
            "final_upright": round(float(tr["upright"][-1]), 3),
        }
    return judge
