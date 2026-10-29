#!/usr/bin/env python3
"""Run one trick many times under randomized conditions and report its success rate.

    python bench/battery.py roulade -n 20
    python bench/battery.py backroll -n 50 --dr train --delay none \
        --policy backroll=policies/backroll_robust.onnx --videos 1

Physics is CPU MuJoCo with the BAM XL330 model (microduck_rl's scripts/infer_policy.py),
not the MuJoCo Warp the policies were trained in: this is a sim2sim test. Set
MICRODUCK_RL to a microduck_rl clone and MICRODUCK_POLICIES to the folder holding the
published policies (alpha_stand.onnx, alpha_walking.onnx, roulade.onnx).

Conditions per trial (--dr):
  nominal  7.4 V battery, 0.1 sag gain, facing +x, exact start pose
  train    the training DR ranges: 6.5-8.2 V, 0.0-0.2 sag, random heading,
           joint noise 0.02 rad, up to 3 deg of start tilt
--push V   additionally shove the trunk at V m/s in a random direction at a
           random time inside --push-window

Writes trials.csv, summary.json and videos to --out (default results/<scenario>/<time>).
Every trial is seeded (seed + trial number), so any trial can be replayed exactly.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from duckbench.scenarios import build_scenarios  # noqa: E402
from duckbench.sim import CONTROL_DT, DuckSim, Policy, ZeroPolicy  # noqa: E402

PUBLISHED = Path(os.environ.get("MICRODUCK_POLICIES", HERE.parent / "external" / "policies"))
DEFAULT_POLICIES = {
    "stand": PUBLISHED / "alpha_stand.onnx",
    "walk": PUBLISHED / "alpha_walking.onnx",
    "roulade": PUBLISHED / "roulade.onnx",
}
RESULTS = HERE.parent / "results"


def sample_conditions(dr, rng, push, push_window, delay="none"):
    """delay: "none" (infer_policy.py's rehearsal), "train" (3-6 physics steps per
    trial, as the training BAM actuator) or an integer number of physics steps."""
    if dr == "nominal":
        c = {"vin": 7.4, "vin_drop_gain": 0.1, "yaw": 0.0, "joint_noise": 0.0, "tilt": 0.0}
    elif dr == "train":
        c = {
            "vin": float(rng.uniform(6.5, 8.2)),
            "vin_drop_gain": float(rng.uniform(0.0, 0.2)),
            "yaw": float(rng.uniform(-math.pi, math.pi)),
            "joint_noise": 0.02,
            "tilt": float(rng.uniform(0.0, math.radians(3.0))),
        }
    else:
        raise ValueError(dr)
    c["push_speed"] = float(push)
    c["push_at_s"] = float(rng.uniform(*push_window)) if push > 0 else None
    c["push_angle"] = float(rng.uniform(0.0, 2 * math.pi)) if push > 0 else None
    # drawn last, so the other conditions are the same with and without delay
    if delay == "none":
        c["delay_steps"] = 0
    elif delay == "train":
        c["delay_steps"] = int(rng.integers(3, 7))
    else:
        c["delay_steps"] = int(delay)
    return c


def run_trial(scn, policies, cond, seed, writer=None, width=640, height=480, fps=25):
    """One trial. With a writer (imageio), frames are streamed to it as they are rendered."""
    rng = np.random.default_rng(seed)
    sim = DuckSim(vin=cond["vin"], vin_drop_gain=cond["vin_drop_gain"], delay_steps=cond.get("delay_steps", 0))
    sim.reset(scn.start, yaw=cond["yaw"], joint_noise=cond["joint_noise"], tilt=cond["tilt"], rng=rng)
    renderer = cam = None
    if writer is not None:
        import mujoco
        renderer = mujoco.Renderer(sim.model, height=height, width=width)
        cam = sim.make_camera(**scn.camera)
        frame_every = max(1, int(round(1.0 / fps / CONTROL_DT)))
    rows = []
    pushed = False
    k = 0
    # per-trial state shared with command callables
    ctx = {"rng": np.random.default_rng(seed + 7), "on_head": False}
    for seg_i, seg in enumerate(scn.segments):
        policy = policies[seg.role]
        for _ in range(int(round(seg.seconds / CONTROL_DT))):
            if cond["push_at_s"] is not None and not pushed and sim.t >= cond["push_at_s"]:
                sim.push(cond["push_speed"], cond["push_angle"])
                pushed = True
            cmd = seg.command_at(sim, ctx)
            sim.step(policy, cmd)
            st = sim.state()
            ctx["on_head"] = st["on_head"]
            st.update(segment=seg_i, cmd_vx=float(cmd[0]), cmd_vy=float(cmd[1]), cmd_wz=float(cmd[2]))
            rows.append(st)
            if renderer is not None and k % frame_every == 0:
                writer.append_data(sim.render(renderer, cam))
            k += 1
    if renderer is not None:
        renderer.close()
    trace = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    return trace, scn.judge(trace)


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def parse_policy_args(items):
    out = {k: Path(v) for k, v in DEFAULT_POLICIES.items()}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--policy expects role=path, got {item!r}")
        role, path = item.split("=", 1)
        out[role] = Path(path)
    return out


def main():
    scenarios = build_scenarios()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", choices=sorted(scenarios))
    ap.add_argument("-n", "--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dr", choices=("nominal", "train"), default="train")
    ap.add_argument("--delay", default="none",
                    help="command delay: none (infer_policy rehearsal), train (3-6 physics steps), or N steps")
    ap.add_argument("--push", type=float, default=0.0, help="push speed in m/s (0 = no push)")
    ap.add_argument("--push-window", type=float, nargs=2, default=(0.5, 3.0), metavar=("FROM", "TO"))
    ap.add_argument("--policy", action="append", metavar="ROLE=PATH",
                    help="policy for a role (stand, walk, roulade, backroll, headstand, standup)")
    ap.add_argument("--handoff-s", type=float, default=1.0,
                    help="trick duration before the stand policy takes over (*_handoff scenarios)")
    ap.add_argument("--videos", type=int, default=0, help="render the first K trials")
    ap.add_argument("--video-failures", type=int, default=1, help="also render the first K failures")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    scn = build_scenarios(handoff_s=args.handoff_s)[args.scenario]
    paths = parse_policy_args(args.policy)
    paths.setdefault("hold", Path(ZeroPolicy.path))
    missing = [r for r in scn.roles if r not in paths]
    if missing:
        raise SystemExit(f"scenario {scn.name} needs --policy {missing[0]}=PATH")
    policies = {}
    for role in scn.roles:
        if role == "hold":
            policies[role] = ZeroPolicy()
            continue
        if not paths[role].is_file():
            raise SystemExit(f"policy file for {role!r} not found: {paths[role]}")
        policies[role] = Policy(paths[role])

    out = args.out or RESULTS / scn.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"scenario : {scn.name} - {scn.description}")
    print(f"policies : " + ", ".join(f"{r}={paths[r].name}" for r in scn.roles))
    print(f"trials   : {args.trials}  dr={args.dr}  push={args.push} m/s  seed={args.seed}")
    print(f"output   : {out}")

    rows, fails = [], []
    t0 = time.time()
    for i in range(args.trials):
        seed = args.seed + i
        rng = np.random.default_rng(seed + 1_000_003)
        cond = sample_conditions(args.dr, rng, args.push, args.push_window, args.delay)
        _, res = run_trial(scn, policies, cond, seed)
        row = {"trial": i, "seed": seed, **cond, **res}
        rows.append(row)
        if not res["success"]:
            fails.append(i)
        mark = "OK" if res["success"] else "NG"
        detail = "  ".join(f"{k}={v}" for k, v in res.items() if k != "success")
        print(f"  [{i + 1:3d}/{args.trials}] {mark}  {detail}", flush=True)

    k = sum(r["success"] for r in rows)
    lo, hi = wilson(k, len(rows))
    elapsed = time.time() - t0

    with open(out / "trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # videos: the first K trials, plus the first K failures (replayed with the same seed)
    to_render = list(range(min(args.videos, len(rows))))
    to_render += [i for i in fails[:args.video_failures] if i not in to_render]
    videos = []
    if to_render:
        import imageio.v2 as imageio
        for i in to_render:
            r = rows[i]
            cond = {key: r[key] for key in ("vin", "vin_drop_gain", "yaw", "joint_noise", "tilt",
                                            "push_speed", "push_at_s", "push_angle", "delay_steps")}
            name = f"{scn.name}_trial{i:03d}_{'OK' if r['success'] else 'NG'}.mp4"
            with imageio.get_writer(str(out / name), fps=25, macro_block_size=8) as writer:
                _, res = run_trial(scn, policies, cond, r["seed"], writer=writer)
            assert res["success"] == r["success"], "replay diverged from the recorded trial"
            videos.append(name)
            print(f"  video: {name}")

    summary = {
        "scenario": scn.name,
        "description": scn.description,
        "policies": {r: str(paths[r]) for r in scn.roles},
        "trials": len(rows),
        "successes": k,
        "success_rate": round(k / len(rows), 4),
        "wilson95": [round(lo, 4), round(hi, 4)],
        "dr": args.dr,
        "delay": args.delay,
        "push_speed": args.push,
        "push_window": list(args.push_window) if args.push > 0 else None,
        "handoff_s": args.handoff_s if "handoff" in scn.name else None,
        "seed": args.seed,
        "failed_trials": fails,
        "videos": videos,
        "wall_time_s": round(elapsed, 1),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{scn.name}: {k}/{len(rows)} succeeded = {100 * k / len(rows):.0f}% "
          f"(95% CI {100 * lo:.0f}-{100 * hi:.0f}%)  in {elapsed:.0f} s")
    if fails:
        print(f"failed trials: {fails}")


if __name__ == "__main__":
    main()
