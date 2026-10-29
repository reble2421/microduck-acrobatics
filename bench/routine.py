#!/usr/bin/env python3
"""Run the trick routine many times and report the success rate of every transition.

    python bench/routine.py -n 50 --delay train \
        --policy backroll=policies/backroll_robust.onnx \
        --policy headstand=policies/headspin_robust.onnx --spin "4:4,-4:4" --headstand-s 3

walk -> stop -> roll -> recover -> headstand -> [spins -> stop] -> release (fall) ->
stand up, each a hot-swap of ONNX policies in CPU MuJoCo + BAM (duckbench.sim), with
the training DR ranges per trial (--dr). A failed check ends the trial, so the table
is a funnel: "succeeded / reached" is the success rate of that transition given that
everything before it worked. Without --policy backroll, the published forward
roulade stands in for the roll.

Writes routine.md, trials.csv, summary.json and videos to --out.
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from battery import DEFAULT_POLICIES, RESULTS, sample_conditions, wilson  # noqa: E402
from duckbench.routine import build_routine  # noqa: E402
from duckbench.sim import CONTROL_DT, DuckSim, Policy, ZeroPolicy  # noqa: E402

FONT = "/mnt/c/Windows/Fonts/BIZ-UDGothicB.ttc"   # BIZ UDGothic (SIL Open Font License)


def run_routine(stages, policies, cond, seed, writer=None, size=(960, 544), trace_out=None):
    """One trial. Returns (results per stage, failed stage name or None).
    trace_out: optional dict, filled with the per-step trace and the stage start indices."""
    rng = np.random.default_rng(seed)
    sim = DuckSim(vin=cond["vin"], vin_drop_gain=cond["vin_drop_gain"], delay_steps=cond.get("delay_steps", 0))
    sim.reset("stand", yaw=cond["yaw"], joint_noise=cond["joint_noise"], tilt=cond["tilt"], rng=rng)
    rows, start, results = [], {}, []
    renderer = cam = draw = None
    if writer is not None:
        import mujoco
        from PIL import Image, ImageDraw, ImageFont
        W, H = size
        sim.model.vis.global_.offwidth = max(W, sim.model.vis.global_.offwidth)    # scene default 640x480
        sim.model.vis.global_.offheight = max(H, sim.model.vis.global_.offheight)
        renderer = mujoco.Renderer(sim.model, height=H, width=W)
        cam = sim.make_camera(distance=1.0, elevation=-15.0, azimuth=90.0)
        font_l, font_s = ImageFont.truetype(FONT, int(H * 0.055)), ImageFont.truetype(FONT, int(H * 0.036))
        look = None

        def draw(frame, stage, t_stage, t):
            img = Image.fromarray(frame)
            dr = ImageDraw.Draw(img, "RGBA")
            pad = int(H * 0.02)
            dr.rectangle([0, 0, W, int(H * 0.09)], fill=(0, 0, 0, 140))
            dr.text((pad, pad), f"{stage.label}", font=font_l, fill=(255, 255, 255))
            dr.text((W - pad, pad), f"t = {t:4.1f} s（等速）", font=font_s, fill=(220, 220, 220), anchor="ra")
            wz = float(stage.cmd()[2])
            if stage.role == "headstand" and wz != 0.0 and len(rows) > 26:
                # spin readout: command vs the head heading's change over the last 0.5 s
                d = np.diff(np.array([r["heading"] for r in rows[-26:]]))
                rate = float(((d + np.pi) % (2 * np.pi) - np.pi).sum()) / (25 * CONTROL_DT)
                txt = f"指令 {abs(wz):.1f} rad/s（{math.degrees(abs(wz)):.0f}°/s）  実際 {abs(rate):.1f} rad/s"
                txt += f"（{rate / wz:.2f}倍）" if t_stage >= 0.8 else "（加速中）"
                bw = dr.textlength(txt, font=font_s) + 2 * pad
                dr.rectangle([W - bw - pad, int(H * 0.11), W - pad, int(H * 0.11) + int(H * 0.06)],
                             fill=(0, 0, 0, 150))
                dr.text((W - 2 * pad, int(H * 0.11) + pad // 2), txt, font=font_s, fill=(255, 255, 255), anchor="ra")
            y = int(H * 0.12)
            for r in results:
                ok = r["ok"]
                mark = "○" if ok else "×"
                col = (120, 230, 120) if ok else (255, 110, 110)
                dr.text((pad, y), f"{mark} {r['label']}", font=font_s, fill=col)
                y += int(H * 0.048)
            return np.asarray(img)

    pushed = False
    k = 0
    for st in stages:
        start[st.name] = len(rows)
        policy = policies[st.role]
        cmd = st.cmd()
        for j in range(int(round(st.seconds / CONTROL_DT))):
            if cond["push_at_s"] is not None and not pushed and sim.t >= cond["push_at_s"]:
                sim.push(cond["push_speed"], cond["push_angle"])
                pushed = True
            sim.step(policy, cmd)
            s = sim.state()
            rows.append(s)
            if renderer is not None and k % 2 == 0:
                target = np.array([s["com_x"], s["com_y"], 0.08])
                look = target if look is None else look + 0.08 * (target - look)
                cam.lookat[:] = look
                renderer.update_scene(sim.data, camera=cam)
                writer.append_data(draw(renderer.render(), st, j * CONTROL_DT, sim.t))
            k += 1
        if trace_out is not None:
            trace_out["rows"], trace_out["start"] = rows, start
        if st.check is None:
            continue
        tr = {key: np.array([r[key] for r in rows]) for key in rows[0]}
        ok, info = st.check(tr, start)
        results.append({"stage": st.name, "label": st.label, "ok": bool(ok), **info})
        if not ok:
            if renderer is not None:     # hold the last frame a moment so the failure is readable
                last = draw(renderer.render(), st, 0.0, sim.t)
                for _ in range(25):
                    writer.append_data(last)
                renderer.close()
            return results, st.name
    if renderer is not None:
        last = draw(renderer.render(), stages[-1], 0.0, sim.t)
        for _ in range(25):
            writer.append_data(last)
        renderer.close()
    return results, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dr", choices=("nominal", "train"), default="train")
    ap.add_argument("--delay", default="none",
                    help="command delay: none (infer_policy rehearsal), train (3-6 physics steps), or N steps")
    ap.add_argument("--policy", action="append", metavar="ROLE=PATH",
                    help="roles: walk, stand, backroll, headstand (defaults: published walk/stand, "
                         "headstand.onnx v1; no backroll = the forward roulade stands in)")
    ap.add_argument("--roll-s", type=float, default=2.0, help="time on the roll policy before the stand takes over")
    ap.add_argument("--recover-s", type=float, default=2.0,
                    help="time the stand policy gets after the roll before the check (rescues bad landings)")
    ap.add_argument("--release-s", type=float, default=1.0, help="go limp after the headstand (0 = straight to stand)")
    ap.add_argument("--headstand-s", type=float, default=6.0)
    ap.add_argument("--spin", default="",
                    help='headspin after the headstand, "wz:seconds,..." (rad/s, + = counter-clockwise from above), '
                         'e.g. "2:4,-2:4"; the headstand role must then be a headspin policy')
    ap.add_argument("--spin-stop-s", type=float, default=2.0, help="zero command after the spins (checked: stopped)")
    ap.add_argument("--videos", type=int, default=1, help="render the first K trials")
    ap.add_argument("--video-size", default="960x544", help="WxH of the videos, e.g. 1280x720")
    ap.add_argument("--video-failures", type=int, default=1, help="also render the first K failed trials")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    paths = {k: Path(v) for k, v in DEFAULT_POLICIES.items()}
    paths["headstand"] = HERE.parent / "policies" / "headspin_robust.onnx"
    for item in args.policy or []:
        role, p = item.split("=", 1)
        paths[role] = Path(p)
    if "backroll" in paths:
        roll_role, roll_dir = "backroll", -1.0
    else:
        roll_role, roll_dir = "roulade", +1.0
        print("note: no --policy backroll given: the published forward roulade stands in for the roll")
    spins = tuple((float(w), float(s)) for w, s in (p.split(":") for p in args.spin.split(","))) if args.spin else ()
    stages = build_routine(roll_role=roll_role, roll_direction=roll_dir, roll_s=args.roll_s,
                           recover_s=args.recover_s, release_s=args.release_s, headstand_s=args.headstand_s,
                           spins=spins, spin_stop_s=args.spin_stop_s)
    roles = {st.role for st in stages}
    policies = {}
    for role in roles:
        if role == "hold":
            policies[role] = ZeroPolicy()
            continue
        if not paths.get(role, Path("")).is_file():
            raise SystemExit(f"policy for {role!r} not found: {paths.get(role)}")
        policies[role] = Policy(paths[role])

    out = args.out or RESULTS / "routine" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print("stages  : " + " -> ".join(f"{st.label}({st.role} {st.seconds:g}s)" for st in stages))
    print("policies: " + ", ".join(f"{r}={paths[r].name}" for r in sorted(roles - {'hold'})))
    print(f"trials  : {args.trials}  dr={args.dr}  seed={args.seed}  -> {out}")

    checked = [st for st in stages if st.check is not None]
    reached = {st.name: 0 for st in checked}
    passed = {st.name: 0 for st in checked}
    rows, fails, conds = [], [], []
    t0 = time.time()
    for i in range(args.trials):
        cond = sample_conditions(args.dr, np.random.default_rng(args.seed + i + 1_000_003), 0.0, (0.5, 3.0),
                                 args.delay)
        conds.append(cond)
        results, failed = run_routine(stages, policies, cond, args.seed + i)
        for r in results:
            reached[r["stage"]] += 1
            passed[r["stage"]] += int(r["ok"])
        if failed:
            fails.append(i)
        row = {"trial": i, "seed": args.seed + i, "failed_at": failed or "", **cond}
        for r in results:
            for key, v in r.items():
                if key not in ("stage", "label"):
                    row[f"{r['stage']}.{key}"] = v
        rows.append(row)
        print(f"  [{i + 1:3d}/{args.trials}] " + " ".join(("OK:" if r["ok"] else "NG:") + r["label"] for r in results),
              flush=True)

    # funnel table
    md = ["| 段階 | たどり着いた | 成功 | 成功率（95%信頼区間） |", "|---|---|---|---|"]
    for st in checked:
        n, k = reached[st.name], passed[st.name]
        lo, hi = wilson(k, n)
        rate = f"{100 * k / n:.0f}% ({100 * lo:.0f}-{100 * hi:.0f}%)" if n else "-"
        md.append(f"| {st.label} | {n} | {k} | {rate} |")
    k_all = passed[checked[-1].name]
    lo, hi = wilson(k_all, args.trials)
    head = (f"# 連続技の成功率（{datetime.now():%Y-%m-%d %H:%M}）\n\n"
            + " → ".join(st.label for st in stages) + f"\n\n{args.trials}回、dr={args.dr}、指令の遅れ={args.delay}。"
            f"最後まで成功: **{k_all}/{args.trials}（{100 * lo:.0f}-{100 * hi:.0f}%）**\n\n")
    (out / "routine.md").write_text(head + "\n".join(md) + "\n", encoding="utf-8")
    keys = sorted({k for r in rows for k in r}, key=lambda s: (s.count("."), s))
    with open(out / "trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    videos = []
    to_render = list(range(min(args.videos, args.trials)))
    to_render += [i for i in fails[:args.video_failures] if i not in to_render]
    if to_render:
        import imageio.v2 as imageio
        for i in to_render:
            name = f"routine_trial{i:03d}_{'NG' if i in fails else 'OK'}.mp4"
            with imageio.get_writer(str(out / name), fps=25, macro_block_size=8, quality=8) as writer:
                run_routine(stages, policies, conds[i], args.seed + i, writer=writer,
                            size=tuple(int(v) for v in args.video_size.split("x")))
            videos.append(name)
            print(f"  video: {name}")

    summary = {"stages": [st.name for st in stages], "policies": {r: str(paths[r]) for r in roles - {"hold"}},
               "trials": args.trials, "completed": k_all, "wilson95": [round(lo, 4), round(hi, 4)],
               "reached": reached, "passed": passed, "failed_trials": fails, "videos": videos,
               "dr": args.dr, "delay": args.delay, "roll_s": args.roll_s, "recover_s": args.recover_s, "release_s": args.release_s, "spin": args.spin,
               "wall_time_s": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + head + "\n".join(md))


if __name__ == "__main__":
    main()
