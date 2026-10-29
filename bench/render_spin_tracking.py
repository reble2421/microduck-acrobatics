#!/usr/bin/env python3
"""Headspin on command, shown as a video with a live chart of commanded vs measured spin rate.

    python bench/render_spin_tracking.py policies/headspin_robust.onnx OUT.mp4 \
        [--script "0:2,3:4,4:4,5:4,0:2.5,-3:4,-4:4,-5:4,0:3"] [--delay 4] [--seed 0]

CPU MuJoCo + BAM (duckbench.sim), starting balanced on the head, real time at 25 fps.
--script: comma-separated wz:seconds steps (rad/s, + = counter-clockwise seen from above).
The measured rate is the head heading's change over the last 0.5 s. The chart at the
bottom draws the command (dashed) and the measured rate (solid) as the clip plays.
Also writes OUT.csv (t, command, measured) for the report.
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("LP_NUM_THREADS", "2")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from battery import sample_conditions  # noqa: E402
from duckbench.scenarios import twist  # noqa: E402
from duckbench.sim import CONTROL_DT, DuckSim, Policy  # noqa: E402

FONT_B = "/mnt/c/Windows/Fonts/BIZ-UDGothicB.ttc"   # BIZ UDGothic (SIL Open Font License)
FONT_R = "/mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc"
W, H = 1280, 720
CMD_COL = (217, 89, 38)      # categorical slot 2, dark-surface step (command)
MEAS_COL = (57, 135, 229)    # categorical slot 1, dark-surface step (measured)
INK, INK2, GRID = (255, 255, 255), (195, 194, 183), (90, 90, 86)
PANEL = (26, 26, 25, 200)
WINDOW_S = 0.5               # measured rate: heading change over this window
SETTLE_S = 0.8               # ratio shown once this long into a new command


def side(wz):
    return "止まれ" if wz == 0 else ("左回り" if wz > 0 else "右回り")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--script", default="0:2,3:4,4:4,5:4,0:2.5,-3:4,-4:4,-5:4,0:3")
    ap.add_argument("--delay", default="4", help="command delay in physics steps (5 ms each), or 'train'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="ヘッドスピン：指令どおりの速さで回る")
    args = ap.parse_args()
    steps = [(float(a), float(b)) for a, b in (s.split(":") for s in args.script.split(","))]
    total = sum(s for _, s in steps)

    cond = sample_conditions("train", np.random.default_rng(args.seed + 1_000_003), 0.0, (0.5, 3.0),
                             args.delay if args.delay == "train" else int(args.delay))
    sim = DuckSim(vin=cond["vin"], vin_drop_gain=cond["vin_drop_gain"], delay_steps=cond["delay_steps"])
    sim.reset("headstand", yaw=cond["yaw"], joint_noise=cond["joint_noise"], tilt=cond["tilt"],
              rng=np.random.default_rng(args.seed))
    policy = Policy(args.onnx)

    sim.model.vis.global_.offwidth = max(W, sim.model.vis.global_.offwidth)    # scene default 640x480
    sim.model.vis.global_.offheight = max(H, sim.model.vis.global_.offheight)
    renderer = mujoco.Renderer(sim.model, height=H, width=W)
    cam = sim.make_camera(distance=0.75, elevation=-22.0, azimuth=135.0)
    f_title, f_big = ImageFont.truetype(FONT_B, 34), ImageFont.truetype(FONT_B, 30)
    f_mid, f_small = ImageFont.truetype(FONT_R, 24), ImageFont.truetype(FONT_R, 19)

    # chart geometry (bottom panel)
    cx0, cx1, cy0, cy1 = 110, W - 40, H - 215, H - 72
    vmax = max(5.5, max(abs(w) for w, _ in steps) + 0.5)
    X = lambda t: cx0 + (cx1 - cx0) * t / total          # noqa: E731
    Y = lambda v: (cy0 + cy1) / 2 - (cy1 - cy0) / 2 * v / vmax   # noqa: E731
    cmd_line = []
    t0 = 0.0
    for w, s in steps:
        cmd_line += [(X(t0), Y(w)), (X(t0 + s), Y(w))]
        t0 += s

    headings, trace = [], []
    look = None
    n_win = int(round(WINDOW_S / CONTROL_DT))
    out_csv = args.out.with_suffix(".csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    k = 0
    with imageio.get_writer(str(args.out), fps=25, macro_block_size=8, quality=8) as writer:
        t_start = 0.0
        for wz, secs in steps:
            for j in range(int(round(secs / CONTROL_DT))):
                sim.step(policy, twist(wz=wz))
                s = sim.state()
                headings.append(s["heading"])
                if len(headings) > n_win:
                    d = np.diff(np.asarray(headings[-n_win - 1:]))
                    rate = float(((d + np.pi) % (2 * np.pi) - np.pi).sum()) / WINDOW_S
                else:
                    rate = 0.0
                trace.append((sim.t, wz, rate, s["on_head"]))
                if k % 2 == 0:
                    target = np.array([s["com_x"], s["com_y"], 0.10])
                    look = target if look is None else look + 0.08 * (target - look)
                    cam.lookat[:] = look
                    renderer.update_scene(sim.data, camera=cam)
                    img = Image.fromarray(renderer.render()).convert("RGBA")
                    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    dr = ImageDraw.Draw(ov)
                    # title bar
                    dr.rectangle([0, 0, W, 58], fill=(0, 0, 0, 150))
                    dr.text((24, 12), args.title, font=f_title, fill=INK)
                    dr.text((W - 24, 18), f"t = {sim.t:4.1f} s（等速）", font=f_mid, fill=INK2, anchor="ra")
                    # readout
                    dr.rounded_rectangle([24, 76, 520, 214], radius=10, fill=PANEL)
                    deg = math.degrees(abs(wz))
                    cmd_txt = f"{side(wz)}  {abs(wz):.1f} rad/s（{deg:.0f}°/s）" if wz else "止まれ"
                    dr.text((44, 88), "指令", font=f_mid, fill=INK2)
                    dr.text((130, 84), cmd_txt, font=f_big, fill=INK)
                    dr.text((44, 136), "実際", font=f_mid, fill=INK2)
                    dr.text((130, 132), f"{abs(rate):.1f} rad/s（{math.degrees(abs(rate)):.0f}°/s）",
                            font=f_big, fill=INK)
                    if wz and j * CONTROL_DT >= SETTLE_S:
                        ratio_txt = f"指令の {rate / wz:.2f} 倍"
                    elif wz:
                        ratio_txt = "加速中"
                    else:
                        ratio_txt = ""
                    dr.text((130, 178), ratio_txt, font=f_mid, fill=INK2)
                    # chart panel
                    dr.rounded_rectangle([24, cy0 - 40, W - 24, H - 16], radius=10, fill=PANEL)
                    for v in (-5, 0, 5):
                        if abs(v) <= vmax:
                            dr.line([(cx0, Y(v)), (cx1, Y(v))], fill=GRID, width=1)
                            dr.text((cx0 - 12, Y(v)), f"{v:+d}" if v else "0", font=f_small, fill=INK2, anchor="rm")
                    dr.text((40, cy0 - 32), "回転の速さ（rad/s、＋は左回り）", font=f_small, fill=INK2)
                    # legend: swatch + white words (identity is not carried by text colour)
                    lx = W - 360
                    for i in range(0, 34, 10):
                        dr.line([(lx + i, cy0 - 21), (lx + i + 6, cy0 - 21)], fill=CMD_COL, width=4)
                    dr.text((lx + 44, cy0 - 21), "指令", font=f_small, fill=INK, anchor="lm")
                    dr.line([(lx + 120, cy0 - 21), (lx + 154, cy0 - 21)], fill=MEAS_COL, width=4)
                    dr.text((lx + 164, cy0 - 21), "実際の回転", font=f_small, fill=INK, anchor="lm")
                    # command: whole plan, dashed
                    for (xa, ya), (xb, yb) in zip(cmd_line[:-1], cmd_line[1:]):
                        n = max(1, int(math.hypot(xb - xa, yb - ya) // 12))
                        for q in range(n):
                            if q % 2 == 0:
                                f0, f1 = q / n, min(1.0, (q + 1) / n)
                                dr.line([(xa + (xb - xa) * f0, ya + (yb - ya) * f0),
                                         (xa + (xb - xa) * f1, ya + (yb - ya) * f1)], fill=CMD_COL, width=4)
                    # measured: up to now, solid
                    pts = [(X(t), Y(r)) for t, _, r, _ in trace[n_win::2]]
                    if len(pts) > 1:
                        dr.line(pts, fill=MEAS_COL, width=4, joint="curve")
                    dr.line([(X(sim.t), cy0 - 6), (X(sim.t), cy1 + 6)], fill=INK2, width=1)
                    dr.text((W - 40, H - 26), "このPCの物理エンジン（MuJoCo＋モーターモデル）／"
                            f"指令の遅れ {5 * cond['delay_steps']} ms", font=f_small, fill=INK2, anchor="rs")
                    writer.append_data(np.asarray(Image.alpha_composite(img, ov).convert("RGB")))
                k += 1
            t_start += secs
    renderer.close()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "command_rad_s", "measured_rad_s", "on_head"])
        w.writerows([(round(t, 3), wz, round(r, 4), int(o)) for t, wz, r, o in trace])
    on = np.mean([o for *_, o in trace])
    print(f"wrote {args.out} ({total:.1f} s) and {out_csv}; on the head {100 * on:.0f}% of the time, "
          f"delay {5 * cond['delay_steps']} ms")


if __name__ == "__main__":
    main()
