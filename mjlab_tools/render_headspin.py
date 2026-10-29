"""Render a headspin ONNX in the mjlab training env (mujoco-warp, CPU) following a spin script.

    cd microduck_rl   (a clone with installers/setup_acrobatics.py applied)
    PYTHONPATH=$PWD/src python ../microduck-acrobatics/mjlab_tools/render_headspin.py \
        ONNX OUT.mp4 --script "0:1.5,2:4,0:2.5,-3:4,0:3" [--title "..."]

--script: comma-separated wz:seconds steps (rad/s; positive = turning left / counter-
clockwise seen from above). Env 0 starts balanced on the head. Every other control
step the state is copied into a plain MuJoCo MjData and rendered at 25 fps, real time,
with the command, the measured yaw rate (0.5 s average) and the COM drift overlaid.
"""
import argparse
import math
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("LP_NUM_THREADS", "2")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mjlab_lib import OnnxMLP, build_env  # noqa: E402

FONT = "/mnt/c/Windows/Fonts/BIZ-UDGothicB.ttc"   # BIZ UDGothic (SIL Open Font License)
ap = argparse.ArgumentParser()
ap.add_argument("onnx")
ap.add_argument("out")
ap.add_argument("--script", default="0:1.5,2:4,0:2.5,-3:4,0:3")
ap.add_argument("--title", default="ヘッドスピン（指令で回る・止まる）")
args = ap.parse_args()
steps = [(float(a), float(b)) for a, b in (s.split(":") for s in args.script.split(","))]
W, H = 960, 544

policy = OnnxMLP(args.onnx)
env, term = build_env(num_envs=1, task="Mjlab-HeadSpin-Flat-MicroDuck")
dt = float(env.step_dt)
m = env.sim.mj_model
d = mujoco.MjData(m)
ren = mujoco.Renderer(m, height=H, width=W, max_geom=4000)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.distance, cam.elevation, cam.azimuth = 0.8, -28.0, 135.0
head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/jaw_soft")
trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base")
font_l, font_s = ImageFont.truetype(FONT, 28), ImageFont.truetype(FONT, 22)


def cmd_text(wz):
    if wz == 0:
        return "指令: 止まれ"
    side = "左回り" if wz > 0 else "右回り"
    return f"指令: {side} {abs(wz):.1f} rad/s（{math.degrees(abs(wz)):.0f}°/s）"


writer = imageio.get_writer(args.out, fps=25, macro_block_size=8, quality=8)
obs, _ = env.reset()
alpha = dt / 0.5
yaw_rate = 0.0
prev_yaw = None
com_trail, turns = [], 0.0
t = 0.0
for wz, secs in steps:
    for k in range(int(round(secs / dt))):
        term.vel_command_b[:] = torch.tensor([0.0, 0.0, wz])
        term.is_standing_env[:] = wz == 0.0
        with torch.no_grad():
            obs, *_ = env.step(policy(obs["actor"]))
        t += dt
        d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
        d.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
        mujoco.mj_forward(m, d)
        R = d.xmat[head].reshape(3, 3)
        yaw = math.atan2(R[1, 2], R[0, 2])
        if prev_yaw is not None:
            dy = (yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
            yaw_rate += alpha * (dy / dt - yaw_rate)
            turns += dy / (2 * math.pi)
        prev_yaw = yaw
        com = d.subtree_com[trunk].copy()
        com_trail.append(com[:2])
        if k % 2:
            continue
        cam.lookat[:] = np.array([com_trail[0][0], com_trail[0][1], 0.07])
        ren.update_scene(d, camera=cam)
        scn = ren.scene
        for (x, y) in com_trail[::3][-300:]:
            if scn.ngeom < scn.maxgeom:
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.003, 0, 0]),
                                    np.array([x, y, 0.003]), np.eye(3).flatten(),
                                    np.array([0.25, 0.55, 1.0, 1.0], dtype=np.float32))
                scn.ngeom += 1
        img = Image.fromarray(ren.render())
        dr = ImageDraw.Draw(img, "RGBA")
        dr.rectangle([0, 0, W, 50], fill=(0, 0, 0, 140))
        dr.text((12, 10), args.title, font=font_l, fill=(255, 255, 255))
        dr.text((W - 12, 14), f"t = {t:4.1f} s（等速）", font=font_s, fill=(220, 220, 220), anchor="ra")
        drift = 1000 * float(np.linalg.norm(com[:2] - com_trail[0]))
        lines = [(cmd_text(wz), (255, 220, 120)),
                 (f"実際の回転  {math.degrees(yaw_rate):+5.0f} °/s（{turns:+.1f} 回転）", (255, 255, 255)),
                 (f"重心の位置のずれ  {drift:4.0f} mm", (120, 190, 255))]
        dr.rectangle([6, 58, 560, 64 + 32 * len(lines)], fill=(0, 0, 0, 120))
        for i, (ln, col) in enumerate(lines):
            dr.text((14, 62 + 32 * i), ln, font=font_s, fill=col)
        dr.rectangle([0, H - 36, W, H], fill=(0, 0, 0, 140))
        dr.text((12, H - 30), "学習時の物理エンジン（mujoco-warp）・揺らぎなしの本番動作 ／ 青い点：重心の軌跡",
                font=font_s, fill=(200, 200, 200))
        writer.append_data(np.asarray(img))
writer.close()
print(f"wrote {args.out}: {t:.1f} s, {turns:+.1f} turns in total")
