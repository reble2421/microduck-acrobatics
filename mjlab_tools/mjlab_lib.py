"""Shared helpers for running exported ONNX policies inside the mjlab training env on CPU."""
import onnx
import torch
from onnx import numpy_helper

import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-HeadSpin-Flat-MicroDuck"


class OnnxMLP:
    """obs -> (obs - mean) / div -> Gemm/ELU x3 -> Gemm, read from the exported graph
    (the layout scripts/export.py writes), evaluated with torch: no onnxruntime needed."""

    def __init__(self, path):
        w = {t.name: torch.tensor(numpy_helper.to_array(t)) for t in onnx.load(path).graph.initializer}
        self.mean = w["obs_normalizer._mean"]
        self.div = w[next(k for k in w if k.startswith("onnx::Div"))]
        self.layers = [(w[f"mlp.{i}.weight"], w[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]

    def __call__(self, obs):
        x = (obs - self.mean) / self.div
        for k, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if k < len(self.layers) - 1:
                x = torch.nn.functional.elu(x)
        return x


def build_env(num_envs=4, task=TASK, delay=None):
    """Training env with headstand spawns only, no time-out, commands under our control.
    Returns (env, twist command term): write term.vel_command_b and clear
    term.is_standing_env before every env.step. delay=(lo, hi) overrides the actuator
    command delay in physics steps (default: the task's own, 3-6 for the base tasks)."""
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    if delay is not None:
        from copy import deepcopy
        from dataclasses import replace
        robot = deepcopy(cfg.scene.entities["robot"])
        acts = tuple(replace(a, delay_min_lag=delay[0], delay_max_lag=delay[1])
                     for a in robot.articulation.actuators)
        robot.articulation = replace(robot.articulation, actuators=acts)
        cfg.scene.entities["robot"] = robot
    cfg.episode_length_s = 1e4
    cfg.curriculum.pop("headstand_spawn_mix", None)
    cfg.curriculum.pop("headstand_command_ranges", None)
    cfg.curriculum.pop("action_rate_weight", None)  # keep weights fixed, irrelevant here
    cfg.events["set_headstand_state"].params.update(headstand_prob=1.0, standing_prob=0.0, midway_prob=0.0)
    cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    return env, env.command_manager.get_term("twist")
