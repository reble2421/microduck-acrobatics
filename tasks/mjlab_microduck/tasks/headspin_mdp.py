"""Headspin: spin on the top of the head at the commanded rate, stop on command.

Task: Mjlab-HeadSpin-Flat-MicroDuck (microduck_headspin_env_cfg.py), built on the
headstand task (same entry shaping, gate, spawns and hold reward; headstand_mdp.py).

State kept on the env object (created lazily, refreshed once per control step):
  env._hs_on, env._hs_ema[:, 2]  headstand gate and head yaw rate, 0.5 s EMA (headstand_mdp)
  env._hsp_com_v                 whole-robot COM velocity, 0.5 s EMA, heading frame
  env._hsp_sup_v                 velocity of the head-top face centre (the support
                                 patch on the floor), 0.5 s EMA, heading frame

Rewards (omega = yaw rate EMA, c = commanded yaw rate):
  headspin_track      1 - |omega - c| / |c|, clipped to [floor, 1], for a spin command.
                      The positive part is gated on the COM staying over the support
                      (its EMA speed below drift_ref): spinning while wandering does
                      not pay. Penalties are never gated.
  spin_when_stopped   |omega| / spin_ref (clip 1) for a stop command: a cost, so
                      spinning down on command is trained.
  still_when_stopped  exp(-|v_support|^2 / lin_std^2 - omega^2 / ang_std^2) for a stop
                      command: stay put on the head.

Commands (HeadSpinCommand): zero_frac exact zero (stop), the rest a pure yaw-rate
command with |wz| in [min_frac, 1] x range and a random sign; no linear command.
"""

import math
from dataclasses import dataclass

import torch

from mjlab_microduck.tasks import headstand_mdp as hs
from mjlab_microduck.tasks import mdp as microduck_mdp

LIN_EPS = 1e-3      # m/s: a linear command below this counts as zero
ANG_EPS = 0.02      # rad/s: a yaw command below this counts as zero
EMA_TAU = 0.5       # s, same averaging as the headstand state
SPIN_REF = 0.4      # rad/s: yaw rate that costs the full spin_when_stopped penalty
DRIFT_REF = 0.03    # m/s: COM speed (0.5 s EMA) that removes all spin pay
# Centre of the flat head-top face in the jaw_soft (head) frame; it sits on the
# floor in the balanced headstand (checked in CPU MuJoCo: z = -1 mm).
FACE_CENTER_HEAD = (0.0407513645315408, 0.0009440486873391888, -0.07720818445096261)


def _split_gate(r: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Gate the pay, never the penalty."""
    return torch.clamp(r, min=0.0) * gate + torch.clamp(r, max=0.0)


# ---- whole-robot COM velocity ------------------------------------------------------

def _com_buffers(env) -> None:
    if hasattr(env, "_hsp_com_ema_w"):
        return
    n, dev = env.num_envs, env.device
    env._hsp_com_prev = torch.zeros(n, 3, device=dev)
    env._hsp_com_ema_w = torch.zeros(n, 2, device=dev)
    env._hsp_com_v = torch.zeros(n, 2, device=dev)       # EMA COM velocity, heading frame
    env._hsp_com_fresh = torch.ones(n, dtype=torch.bool, device=dev)
    env._hsp_com_last_step = -1


def robot_com(env) -> torch.Tensor:
    """(N, 3) world position of the whole-robot COM (includes mass DR)."""
    asset = env.scene["robot"]
    return env.sim.data.subtree_com[:, asset.indexing.root_body_id]


def update_com_velocity(env) -> None:
    """Refresh the headstand state and the COM velocity EMA once per control step."""
    hs.update_headstand(env)            # env._hs_on, env._hs_ema[:, 2] (yaw rate EMA)
    _com_buffers(env)
    step = int(env.common_step_counter)
    if step == env._hsp_com_last_step:
        return
    env._hsp_com_last_step = step
    com = torch.nan_to_num(robot_com(env).clone(), nan=0.0)
    fresh = env._hsp_com_fresh.unsqueeze(-1)
    prev = torch.where(fresh, com, env._hsp_com_prev)
    v = (com[:, :2] - prev[:, :2]) / float(env.step_dt)
    alpha = min(1.0, float(env.step_dt) / EMA_TAU)
    ema = env._hsp_com_ema_w + alpha * (v - env._hsp_com_ema_w)
    env._hsp_com_ema_w = torch.where(fresh, torch.zeros_like(ema), ema)
    env._hsp_com_prev = com
    env._hsp_com_fresh = torch.zeros_like(env._hsp_com_fresh)
    asset = env.scene["robot"]
    hq = asset.data.body_link_quat_w[:, env._hs_head_id]
    yaw = torch.nan_to_num(hs.head_heading(hq), nan=0.0)
    env._hsp_com_v = hs.to_heading_frame(env._hsp_com_ema_w, yaw)


def reset_com_velocity(env, env_ids: torch.Tensor) -> None:
    """Reset event: the next update starts the COM difference afresh (no spike)."""
    _com_buffers(env)
    if env_ids is None:
        env._hsp_com_fresh[:] = True
    else:
        env._hsp_com_fresh[env_ids.to(env.device, dtype=torch.long)] = True


# ---- velocity of the support patch (head-top face centre) ------------------------------

def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (N, 3) by quaternions q (N, 4) in (w, x, y, z) order."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _support_buffers(env) -> None:
    if hasattr(env, "_hsp_sup_ema_w"):
        return
    n, dev = env.num_envs, env.device
    env._hsp_sup_ema_w = torch.zeros(n, 2, device=dev)
    env._hsp_sup_v = torch.zeros(n, 2, device=dev)        # support velocity EMA, heading frame
    env._hsp_sup_fresh = torch.ones(n, dtype=torch.bool, device=dev)
    env._hsp_sup_last_step = -1


def support_velocity_w(env) -> torch.Tensor:
    """(N, 3) world velocity of the head-top face centre."""
    asset = env.scene["robot"]
    hid = env._hs_head_id
    q = asset.data.body_link_quat_w[:, hid]
    p = torch.tensor(FACE_CENTER_HEAD, device=env.device, dtype=q.dtype).expand(q.shape[0], 3)
    r = _quat_apply(q, p)
    return asset.data.body_link_lin_vel_w[:, hid] + torch.cross(asset.data.body_link_ang_vel_w[:, hid], r, dim=-1)


def update_support_velocity(env) -> None:
    """Refresh the headstand state and the support velocity EMA once per control step."""
    hs.update_headstand(env)            # env._hs_on, env._hs_ema[:, 2] (yaw rate), env._hs_head_id
    _support_buffers(env)
    step = int(env.common_step_counter)
    if step == env._hsp_sup_last_step:
        return
    env._hsp_sup_last_step = step
    v = torch.nan_to_num(support_velocity_w(env)[:, :2], nan=0.0)
    alpha = min(1.0, float(env.step_dt) / EMA_TAU)
    fresh = env._hsp_sup_fresh.unsqueeze(-1)
    prev = torch.where(fresh, torch.zeros_like(v), env._hsp_sup_ema_w)
    env._hsp_sup_ema_w = prev + alpha * (v - prev)
    env._hsp_sup_fresh = torch.zeros_like(env._hsp_sup_fresh)
    asset = env.scene["robot"]
    yaw = torch.nan_to_num(hs.head_heading(asset.data.body_link_quat_w[:, env._hs_head_id]), nan=0.0)
    env._hsp_sup_v = hs.to_heading_frame(env._hsp_sup_ema_w, yaw)


def reset_support_velocity(env, env_ids: torch.Tensor) -> None:
    """Reset event: start the support velocity EMA afresh."""
    _support_buffers(env)
    if env_ids is None:
        env._hsp_sup_fresh[:] = True
    else:
        env._hsp_sup_fresh[env_ids.to(env.device, dtype=torch.long)] = True


# ---- rewards ----------------------------------------------------------------------

def headspin_track(env, command_name: str = "twist", floor: float = -0.25,
                   drift_ref: float = DRIFT_REF) -> torch.Tensor:
    """1 - |wz - c| / |c| for a spin command, pay gated on the COM staying over the support."""
    update_com_velocity(env)
    wz = env.command_manager.get_command(command_name)[:, 2]
    mag = torch.abs(wz)
    err = torch.abs(env._hs_ema[:, 2] - wz) / torch.clamp(mag, min=ANG_EPS)
    r = torch.clamp(1.0 - err, min=floor, max=1.0)
    gate = torch.clamp(1.0 - torch.linalg.norm(env._hsp_com_v, dim=-1) / drift_ref, min=0.0, max=1.0)
    return env._hs_on.float() * (mag > ANG_EPS).float() * _split_gate(r, gate)


def spin_when_stopped(env, command_name: str = "twist", spin_ref: float = SPIN_REF) -> torch.Tensor:
    """|wz| / spin_ref clipped to [0, 1] when no spin is commanded.
    A cost (>= 0): give it a NEGATIVE weight."""
    update_com_velocity(env)
    c = env.command_manager.get_command(command_name)
    no_turn = torch.abs(c[:, 2]) <= ANG_EPS
    cost = torch.clamp(torch.abs(env._hs_ema[:, 2]) / spin_ref, min=0.0, max=1.0)
    return env._hs_on.float() * no_turn.float() * cost


def still_when_stopped(env, command_name: str = "twist", lin_std: float = 0.004,
                       ang_std: float = 0.1) -> torch.Tensor:
    """Stay put for an exact-zero command: support speed and yaw rate near zero."""
    update_support_velocity(env)
    c = env.command_manager.get_command(command_name)
    zero = (torch.linalg.norm(c[:, :2], dim=-1) <= LIN_EPS) & (torch.abs(c[:, 2]) <= ANG_EPS)
    err = (env._hsp_sup_v ** 2).sum(dim=-1) / (lin_std * lin_std) + env._hs_ema[:, 2] ** 2 / (ang_std * ang_std)
    return env._hs_on.float() * zero.float() * torch.exp(-err)


# ---- commands ---------------------------------------------------------------------

def sample_spin_rates(n: int, ang_max: float, min_frac: float, device) -> torch.Tensor:
    """(n,) yaw-rate commands: |wz| uniform in [min_frac, 1] x ang_max, random sign."""
    mag = (min_frac + (1.0 - min_frac) * torch.rand(n, device=device)) * ang_max
    sign = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    return sign * mag


class HeadSpinCommand(microduck_mdp.VelocityCommandCommandOnly):
    """Two buckets per resample: exact zero (zero_frac) or a pure spin command."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        zero = torch.rand(n, device=self.device) < self.cfg.zero_frac
        ang_max = max(abs(self.cfg.ranges.ang_vel_z[0]), abs(self.cfg.ranges.ang_vel_z[1]))
        self.vel_command_b[env_ids, 0:2] = 0.0
        self.vel_command_b[env_ids, 2] = sample_spin_rates(n, ang_max, self.cfg.min_frac, self.device)
        self.is_standing_env[env_ids] = zero          # standing envs get an exact-zero command
        self.vel_command_b[env_ids[zero]] = 0.0
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]


@dataclass(kw_only=True)
class HeadSpinCommandCfg(microduck_mdp.VelocityCommandCommandOnlyCfg):
    zero_frac: float = 0.25
    min_frac: float = 0.5

    def build(self, env) -> "HeadSpinCommand":
        return HeadSpinCommand(self, env)


def deg_s(rad_s: float) -> float:
    return math.degrees(rad_s)
