"""Run a roll ONNX (backroll / roulade) inside its mjlab TRAINING env (mujoco-warp, CPU)
from standing starts, and measure how clean the roll is - the same features as the
CPU-MuJoCo bench uses, so the two engines can be compared directly.

    cd microduck_rl   (a clone with installers/setup_acrobatics.py applied)
    PYTHONPATH=$PWD/src python ../microduck-acrobatics/mjlab_tools/roll_in_mjlab.py \
        ONNX Mjlab-BackRoll-Flat-MicroDuck -1 [N] [seconds]

Per env: net rotation about the body pitch axis (deg, in the roll direction), time
spent lying (|upright| < 0.5 and trunk below 8 cm) after first being inverted, time
from inverted to standing, yaw change start->end, and whether it ends standing.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mjlab_lib import OnnxMLP  # noqa: E402

import mjlab_microduck.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

ONNX, TASK, DIRECTION = sys.argv[1], sys.argv[2], float(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 8
SECONDS = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0

cfg = load_env_cfg(TASK)
cfg.scene.num_envs = N
cfg.episode_length_s = 1e4
cfg.curriculum.pop("roulade_spawn_mix", None)
cfg.events["set_roulade_state"].params.update(standing_prob=1.0, midroll_prob=0.0)
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
policy = OnnxMLP(ONNX)
asset = env.scene["robot"]
dt = float(env.step_dt)


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


obs, _ = env.reset()
origin_z = env.scene.terrain.env_origins[:, 2].cpu()
ups, zs, prs, yaws = [], [], [], []
alive = torch.ones(N, dtype=torch.bool)
for k in range(int(round(SECONDS / dt))):
    with torch.no_grad():
        obs, _, term, trunc, _ = env.step(policy(obs["actor"]))
    alive &= ~(term | trunc).cpu()
    d = asset.data
    ups.append(-d.projected_gravity_b[:, 2].cpu())
    zs.append(d.root_link_pos_w[:, 2].cpu() - origin_z)
    prs.append(d.root_link_ang_vel_b[:, 1].cpu())
    yaws.append(yaw_of(d.root_link_quat_w).cpu())
up, z, pr, yw = (torch.stack(v) for v in (ups, zs, prs, yaws))   # (T, N)

print(f"{os.path.basename(ONNX)} in {TASK} (mujoco-warp, CPU), {N} standing starts, {SECONDS:g} s")
print(f"{'env':>3s} {'body_net':>8s} {'inv':>4s} {'lying_s':>7s} {'t_land':>6s} {'yaw':>6s} end")
clean = 0
for j in range(N):
    body = torch.cumsum(DIRECTION * pr[:, j], 0) * dt
    inv = torch.nonzero(up[:, j] < -0.5).flatten()
    stand = (z[:, j] >= 0.10) & (up[:, j] >= 0.9)
    lying, t_land = 0.0, None
    if len(inv):
        i0 = int(inv[0])
        lying = float(((up[i0:, j].abs() < 0.5) & (z[i0:, j] < 0.08)).sum()) * dt
        st = torch.nonzero(stand[i0:]).flatten()
        t_land = float(st[0]) * dt if len(st) else None
    yaw = math.degrees((float(yw[-1, j] - yw[0, j]) + math.pi) % (2 * math.pi) - math.pi)
    end = bool(stand[-25:].all())
    is_clean = math.degrees(float(body[-1])) >= 300 and lying <= 0.4 and abs(yaw) <= 90 and end
    clean += int(is_clean)
    tl = "-" if t_land is None else f"{t_land:.2f}"
    print(f"{j:3d} {math.degrees(float(body[-1])):8.0f} {str(bool(len(inv)))[0]:>4s} {lying:7.2f} {tl:>6s} "
          f"{yaw:6.0f} {str(end)[0]}{'  clean' if is_clean else ''}")
print(f"clean rolls (net >= 300 deg, lying <= 0.4 s, |yaw| <= 90 deg, ends standing): {clean}/{N}; "
      f"envs never reset: {int(alive.sum())}/{N}")
