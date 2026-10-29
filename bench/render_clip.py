#!/usr/bin/env python3
"""Video clips for the final video: one or several robots side by side, same timeline.

    python bench/render_clip.py PRESET OUT.mp4

PRESET is one of the clips defined in PRESETS below (headspin_solo, backroll_compare,
learning_progress, delay_compare). Every panel is its own CPU MuJoCo + BAM simulation
(duckbench.sim) with the bench's trial conditions for --seed (the same draw as
battery.py's trial number `seed`), so a panel reproduces that bench trial exactly.
Panels share the command timeline; they differ in policy or command delay. Real time,
25 fps, 1280x720.
"""

import argparse
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

from battery import DEFAULT_POLICIES, sample_conditions  # noqa: E402
from duckbench.scenarios import make_roll_judge, twist  # noqa: E402
from duckbench.sim import CONTROL_DT, DuckSim, Policy  # noqa: E402

FONT_B = "/mnt/c/Windows/Fonts/BIZ-UDGothicB.ttc"   # BIZ UDGothic (SIL Open Font License)
FONT_R = "/mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc"
W, H = 1280, 720
POL = HERE.parent / "policies"
HEADSPIN = POL / "headspin_robust.onnx"
INK, INK2 = (255, 255, 255), (205, 204, 196)
OK_COL, NG_COL = (120, 230, 120), (255, 120, 110)

# segment: (role, seconds, wz rad/s, label shown under the title)
PRESETS = {
    "headspin_solo": dict(
        title="ヘッドスピン：指令で回り、指令で止まる", start="headstand", seed=0, layout=(1, 1),
        camera=dict(distance=0.55, elevation=-16.0, azimuth=135.0), spin_readout=True,
        panels=[dict(label="", policies={"headstand": HEADSPIN}, delay=4)],
        segments=[("headstand", 1.5, 0.0, "逆立ち"), ("headstand", 5.0, 4.0, "左回り 毎秒4rad（229°/s）"),
                  ("headstand", 2.0, 0.0, "止まれ"), ("headstand", 5.0, -4.0, "右回り 毎秒4rad（229°/s）"),
                  ("headstand", 2.5, 0.0, "止まれ")],
        footer="このPCの物理エンジン（MuJoCo＋モーターモデル）／指令の遅れ 20ms"),
    "backroll_compare": dict(
        title="後転：遅れなしの条件で、改善前と改善後を比べる", start="stand", seed=3, layout=(1, 2),
        camera=dict(distance=0.8, elevation=-8.0, azimuth=90.0), roll_judge=True,
        panels=[dict(label="改善前（遅れ15〜30msで学習）", policies={"backroll": POL / "backroll.onnx"}, delay=0),
                dict(label="改善後（遅れ0〜30msで学習）",
                     policies={"backroll": POL / "backroll_robust.onnx"}, delay=0)],
        segments=[("backroll", 5.0, 0.0, "後転（同じ初期条件、指令の遅れなし）")],
        footer="このPCの物理エンジン（MuJoCo＋モーターモデル）／評価の第3回と同じ条件"),
    "learning_progress": dict(
        title="学習の進み方：同じ指令（左回り 毎秒3rad）を出したとき", start="headstand", seed=0, layout=(2, 2),
        camera=dict(distance=0.6, elevation=-18.0, azimuth=135.0), spin_readout=True,
        panels=[dict(label="学習前（逆立ち）", policies={"headstand": POL / "headstand.onnx"},
                     delay=4),
                dict(label="1,000反復", policies={"headstand": POL / "checkpoints" / "headspin_it1000.onnx"},
                     delay=4),
                dict(label="2,000反復", policies={"headstand": POL / "checkpoints" / "headspin_it2000.onnx"},
                     delay=4),
                dict(label="最終（5,000反復）", policies={"headstand": HEADSPIN}, delay=4)],
        segments=[("headstand", 1.0, 0.0, "逆立ち"), ("headstand", 7.0, 3.0, "左回り 毎秒3rad（172°/s）")],
        footer="このPCの物理エンジン（MuJoCo＋モーターモデル）／指令の遅れ 20ms／反復＝ヘッドスピンの学習の回数"),
    "delay_compare": dict(
        title="弱点：モーター指令の遅れがないと回れない", start="headstand", seed=0, layout=(1, 2),
        camera=dict(distance=0.6, elevation=-16.0, azimuth=135.0), spin_readout=True,
        panels=[dict(label="指令の遅れ 20ms", policies={"headstand": HEADSPIN}, delay=4),
                dict(label="指令の遅れ なし", policies={"headstand": HEADSPIN}, delay=0)],
        segments=[("headstand", 1.0, 0.0, "逆立ち"), ("headstand", 6.0, 4.0, "左回り 毎秒4rad（229°/s）"),
                  ("headstand", 2.0, 0.0, "止まれ")],
        footer="このPCの物理エンジン（MuJoCo＋モーターモデル）／同じ方策、同じ初期条件"),
}


class Panel:
    def __init__(self, spec, preset, size):
        cond = sample_conditions("train", np.random.default_rng(preset["seed"] + 1_000_003), 0.0, (0.5, 3.0),
                                 int(spec["delay"]))
        self.sim = DuckSim(vin=cond["vin"], vin_drop_gain=cond["vin_drop_gain"], delay_steps=cond["delay_steps"])
        self.sim.reset(preset["start"], yaw=cond["yaw"], joint_noise=cond["joint_noise"], tilt=cond["tilt"],
                       rng=np.random.default_rng(preset["seed"]))
        roles = {**{r: Path(p) for r, p in DEFAULT_POLICIES.items()}, **spec["policies"]}
        self.policies = {r: Policy(roles[r]) for r in {s[0] for s in preset["segments"]}}
        self.label = spec["label"]
        w, h = size
        m = self.sim.model
        m.vis.global_.offwidth, m.vis.global_.offheight = max(w, m.vis.global_.offwidth), max(h, m.vis.global_.offheight)
        self.renderer = mujoco.Renderer(m, height=h, width=w)
        self.cam = self.sim.make_camera(**preset["camera"])
        self.look = None
        self.rows = []

    def step(self, role, wz):
        self.sim.step(self.policies[role], twist(wz=wz))
        self.rows.append(self.sim.state())

    def rate(self, window=25):
        if len(self.rows) <= window:
            return 0.0
        d = np.diff(np.array([r["heading"] for r in self.rows[-window - 1:]]))
        return float(((d + np.pi) % (2 * np.pi) - np.pi).sum()) / (window * CONTROL_DT)

    def frame(self):
        s = self.rows[-1]
        target = np.array([s["com_x"], s["com_y"], 0.10])
        self.look = target if self.look is None else self.look + 0.08 * (target - self.look)
        self.cam.lookat[:] = self.look
        self.renderer.update_scene(self.sim.data, camera=self.cam)
        return Image.fromarray(self.renderer.render()).convert("RGBA")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    pr = PRESETS[args.preset]
    rows_n, cols_n = pr["layout"]
    top, foot = 92, 30                            # title bar + segment line / footer strip
    pw, ph = W // cols_n, (H - top - foot) // rows_n
    panels = [Panel(spec, pr, (pw, ph)) for spec in pr["panels"]]
    f_title, f_seg = ImageFont.truetype(FONT_B, 32), ImageFont.truetype(FONT_R, 24)
    f_lab, f_small = ImageFont.truetype(FONT_B, 24 if rows_n == 1 else 21), ImageFont.truetype(FONT_R, 18)
    judge = make_roll_judge(-1) if pr.get("roll_judge") else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    k, t = 0, 0.0
    verdicts = [None] * len(panels)
    with imageio.get_writer(str(args.out), fps=25, macro_block_size=8, quality=8) as writer:

        def compose(seg_label, final=False):
            canvas = Image.new("RGBA", (W, H), (20, 22, 28, 255))
            for i, p in enumerate(panels):
                img = p.frame()
                ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
                dr = ImageDraw.Draw(ov)
                if p.label:
                    tw = dr.textlength(p.label, font=f_lab)
                    dr.rounded_rectangle([10, 10, 30 + tw, 50], radius=8, fill=(0, 0, 0, 150))
                    dr.text((20, 16), p.label, font=f_lab, fill=INK)
                if pr.get("spin_readout"):
                    txt = f"回転 {abs(p.rate()):.1f} rad/s（{math.degrees(abs(p.rate())):.0f}°/s）"
                    tw = dr.textlength(txt, font=f_lab)
                    dr.rounded_rectangle([10, ph - 52, 30 + tw, ph - 12], radius=8, fill=(0, 0, 0, 150))
                    dr.text((20, ph - 46), txt, font=f_lab, fill=INK)
                if final and verdicts[i] is not None:
                    ok, txt = verdicts[i]
                    tw = dr.textlength(txt, font=f_title)
                    dr.rounded_rectangle([(pw - tw) / 2 - 18, ph - 76, (pw + tw) / 2 + 18, ph - 22], radius=10,
                                         fill=(0, 0, 0, 170))
                    dr.text((pw / 2, ph - 49), txt, font=f_title, fill=OK_COL if ok else NG_COL, anchor="mm")
                img = Image.alpha_composite(img, ov)
                r, c = divmod(i, cols_n)
                canvas.paste(img, (c * pw, top + r * ph))
            dr = ImageDraw.Draw(canvas)
            for c in range(1, cols_n):                       # thin dividers between panels
                dr.line([(c * pw, top), (c * pw, H - foot)], fill=(20, 22, 28), width=4)
            for r in range(1, rows_n):
                dr.line([(0, top + r * ph), (W, top + r * ph)], fill=(20, 22, 28), width=4)
            dr.text((22, 10), pr["title"], font=f_title, fill=INK)
            dr.text((W - 22, 16), f"t = {t:4.1f} s（等速）", font=f_seg, fill=INK2, anchor="ra")
            dr.text((22, 54), f"指令：{seg_label}", font=f_seg, fill=INK2)
            dr.text((W - 14, H - foot / 2), pr["footer"], font=f_small, fill=INK2, anchor="rm")
            return np.asarray(canvas.convert("RGB"))

        for role, secs, wz, seg_label in pr["segments"]:
            for _ in range(int(round(secs / CONTROL_DT))):
                for p in panels:
                    p.step(role, wz)
                t += CONTROL_DT
                if k % 2 == 0:
                    writer.append_data(compose(seg_label))
                k += 1
        if judge is not None:
            for i, p in enumerate(panels):
                tr = {key: np.array([r[key] for r in p.rows]) for key in p.rows[0]}
                res = judge(tr)
                land = res.get("land_s")
                if res.get("clean"):
                    verdicts[i] = (True, f"○ 1回転で着地（{land:.1f}秒）")
                elif res.get("roll_deg", 0.0) > 450.0:
                    verdicts[i] = (False, f"× 回りすぎ（{res['roll_deg']:.0f}°）、着地{land:.1f}秒" if land
                                   else f"× 回りすぎ（{res['roll_deg']:.0f}°）")
                elif res.get("end_standing"):
                    verdicts[i] = (False, "△ 立つが、着地が乱れる")
                else:
                    verdicts[i] = (False, "× 立ち上がれない")
                print(f"panel {i} ({p.label}): {res}")
        last = compose(pr["segments"][-1][3], final=True)
        for _ in range(50 if judge is not None else 12):        # hold the end (2 s with verdicts)
            writer.append_data(last)
    for p in panels:
        p.renderer.close()
        on = np.mean([r["on_head"] for r in p.rows])
        print(f"{p.label or 'panel'}: final rate {p.rate():+.2f} rad/s, on head {100 * on:.0f}%")
    print(f"wrote {args.out} ({t:.1f} s + end hold)")


if __name__ == "__main__":
    main()
