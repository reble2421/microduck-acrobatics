"""CPU MuJoCo + BAM rehearsal of the Microduck, for judging exported ONNX policies.

Everything that has to match deployment is taken from microduck_rl's own
scripts/infer_policy.py: the scene, the BAM XL330 actuator model, DEFAULT_POSE
and the 50 Hz / 4-substep control loop. The 61-D observation is rebuilt here
without PolicyInference's keyboard state machine; tests/test_sim.py checks that
it is identical to PolicyInference.get_observations().

This is a different physics engine from the MuJoCo Warp the policies are
trained in, so a success rate measured here is a sim2sim transfer result.
"""

import contextlib
import io
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("MICRODUCK_RL", Path(__file__).resolve().parents[2] / "external" / "microduck_rl"))
os.environ.setdefault("MUJOCO_GL", "osmesa")  # must be set before mujoco is imported
os.environ.setdefault("LP_NUM_THREADS", "2")  # software renderer threads: keep the AVD calm

import mujoco  # noqa: E402
import onnxruntime as ort  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
import infer_policy as ip  # noqa: E402

TIMESTEP = 0.005
DECIMATION = 4
CONTROL_DT = TIMESTEP * DECIMATION  # 50 Hz
DEFAULT_POSE = ip.DEFAULT_POSE.astype(np.float32)
STAND_Z = 0.125

# Headstand balance pose. Must match headstand_mdp.py (tools/setup_headstand.py).
HEADSTAND_JOINT_POS = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.453,
    -1.4312, 0.5236, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.453,
], dtype=np.float32)
HEADSTAND_QUAT = np.array([0.81159, -0.00337, 0.58422, 0.0])
HEADSTAND_Z = 0.1301
HEAD_BODY = "jaw_soft"
HEAD_TOP_AXIS = np.array([0.882, 0.0, 0.471])
TOP_DOWN_MIN = 0.5
FOOT_BODIES = ("ankle_left", "ankle_right")

# Lying starts, as in mdp.set_random_ground_state (dropped from 0.20-0.25 m).
_S = 2.0 ** -0.5
FACE_DOWN_QUAT = np.array([_S, 0.0, _S, 0.0])   # +90 deg pitch
FACE_UP_QUAT = np.array([_S, 0.0, -_S, 0.0])    # -90 deg pitch
PRONE_Z = 0.22

PUSH_MAX = 1.0  # m/s, infer_policy.py's random_push (end of the velstand curriculum)


@contextlib.contextmanager
def quiet(enabled=True):
    """Silence infer_policy's prints while building things."""
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def tilt_quat(angle, direction):
    """Rotation by `angle` about the horizontal axis at `direction` (rad)."""
    s = math.sin(angle / 2)
    return np.array([math.cos(angle / 2), s * math.cos(direction), s * math.sin(direction), 0.0])


class Policy:
    """One exported ONNX policy ([1, 61] -> [1, 14])."""

    def __init__(self, path, action_scale=1.0):
        self.path = str(path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1   # a 61->14 MLP gains nothing from threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(self.path, sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.action_scale = action_scale
        shape_in = self.session.get_inputs()[0].shape
        if shape_in[-1] != 61:
            raise ValueError(f"{path}: expected a 61-D input, got {shape_in}")

    def __call__(self, obs):
        out = self.session.run([self.output_name], {self.input_name: obs.reshape(1, -1)})[0]
        return out.squeeze(0).astype(np.float32)


class ZeroPolicy:
    """Action 0 = hold DEFAULT_POSE: lets the robot settle where it lies."""

    path = "(hold the default pose)"
    action_scale = 1.0

    def __call__(self, obs):
        return np.zeros(14, dtype=np.float32)


class DuckSim:
    """One robot in the default scene, driven by ONNX policies at 50 Hz.

    vin / vin_drop_gain: battery voltage and load-dependent sag of the BAM
    model (training randomizes 6.5-8.2 V and 0.0-0.2). They are baked into the
    compiled model, so a new DuckSim is built per voltage.
    """

    def __init__(self, vin=7.4, vin_drop_gain=0.1, scene=None, verbose=False, delay_steps=0):
        """delay_steps: command delay in PHYSICS steps (5 ms). Training's BAM actuator
        delays every command by 3-6 physics steps (microduck_constants.py,
        delay_min_lag=3 / delay_max_lag=6); infer_policy.py rehearses with none."""
        self.delay_steps = int(delay_steps)
        self._pending = []       # (physics step at which it applies, target)
        self._substep = 0
        xml = str(REPO / (scene or ip.MICRODUCK_XML))
        with quiet(not verbose):
            bam_model = ip.load_bam_model(ip.BAM_KP_FW, vin, None)
            self.model, self.data, self.bam, _ = ip.load_mujoco_with_bam(
                xml, bam_model, TIMESTEP, vin_drop_gain, ip.BAM_VIN_MIN)
        m = self.model
        self.vin, self.vin_drop_gain = vin, vin_drop_gain
        self.n_joints = m.nu
        self.qpos_idx = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        self.qvel_idx = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        free = m.joint("trunk_base_freejoint")
        self.free_q = int(free.qposadr[0])
        self.free_v = int(free.dofadr[0])
        self.trunk = m.body("trunk_base").id
        self.head = m.body(HEAD_BODY).id
        self.feet = {m.body(b).id for b in FOOT_BODIES}
        self.robot_bodies = {b for b in range(1, m.nbody) if m.body_rootid[b] == self.trunk}
        gyro = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.gyro_adr = int(m.sensor_adr[gyro])
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)
        self.t = 0.0

    # ---- start states ---------------------------------------------------------

    def reset(self, start="stand", yaw=0.0, joint_noise=0.0, tilt=0.0, rng=None):
        """Place the robot. start: stand | headstand | face_down | face_up."""
        rng = rng or np.random.default_rng(0)
        d = self.data
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        if start == "stand":
            z, base, pose = STAND_Z, np.array([1.0, 0, 0, 0]), DEFAULT_POSE
        elif start == "headstand":
            z, base, pose = HEADSTAND_Z + 0.006, HEADSTAND_QUAT, HEADSTAND_JOINT_POS
        elif start == "face_down":
            z, base, pose = PRONE_Z, FACE_DOWN_QUAT, DEFAULT_POSE
        elif start == "face_up":
            z, base, pose = PRONE_Z, FACE_UP_QUAT, DEFAULT_POSE
        else:
            raise ValueError(f"unknown start state: {start}")
        quat = base
        if tilt > 0.0:
            quat = quat_mul(tilt_quat(tilt, rng.uniform(0, 2 * math.pi)), quat)
        quat = quat_mul(yaw_quat(yaw), quat)
        pose = np.asarray(pose, dtype=np.float32)
        if joint_noise > 0.0:
            pose = pose + rng.normal(0.0, joint_noise, size=pose.shape).astype(np.float32)
        d.qpos[self.free_q:self.free_q + 3] = [0.0, 0.0, z]
        d.qpos[self.free_q + 3:self.free_q + 7] = quat / np.linalg.norm(quat)
        for i, adr in enumerate(self.qpos_idx):
            d.qpos[adr] = pose[i]
        self.bam.reset(d.qpos)
        self.bam.q_target[:] = pose
        self.last_action[:] = 0.0
        self._pending, self._substep = [], 0
        self.t = 0.0
        mujoco.mj_forward(self.model, d)

    # ---- observation / control --------------------------------------------------

    def projected_gravity(self):
        # Same float32 formula as PolicyInference.quat_rotate_inverse.
        q = self.data.xquat[self.trunk].astype(np.float32)
        w, xyz = q[0], q[1:4]
        g = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        t = np.cross(xyz, g) * 2
        return g - w * t + np.cross(xyz, t)

    def observation(self, command):
        d = self.data
        return np.concatenate([
            d.sensordata[self.gyro_adr:self.gyro_adr + 3].astype(np.float32),
            self.projected_gravity(),
            d.qpos[self.qpos_idx].astype(np.float32) - DEFAULT_POSE,
            d.qvel[self.qvel_idx].astype(np.float32),
            self.last_action,
            np.asarray(command, dtype=np.float32),
        ]).astype(np.float32)

    def step(self, policy, command):
        """One 50 Hz control step. last_action carries over across policy
        swaps, exactly like the runtime's hot-swap (and PolicyInference)."""
        action = policy(self.observation(command))
        self.last_action = action
        target = DEFAULT_POSE + action * policy.action_scale
        if self.delay_steps <= 0:
            self.bam.q_target[:] = target
        else:
            self._pending.append((self._substep + self.delay_steps, target))
        for _ in range(DECIMATION):
            while self._pending and self._pending[0][0] <= self._substep:
                self.bam.q_target[:] = self._pending.pop(0)[1]
            self.bam.update()
            mujoco.mj_step(self.model, self.data)
            self._substep += 1
        self.t += CONTROL_DT
        return action

    def push(self, speed=PUSH_MAX, angle=0.0):
        """Overwrite the trunk's world xy velocity (a shove)."""
        self.data.qvel[self.free_v + 0] = speed * math.cos(angle)
        self.data.qvel[self.free_v + 1] = speed * math.sin(angle)

    # ---- state for judging -----------------------------------------------------------

    def floor_contacts(self):
        """Robot bodies touching the floor (body 0 = world)."""
        touching = set()
        m, d = self.model, self.data
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 == 0 and b2 in self.robot_bodies:
                touching.add(int(b2))
            elif b2 == 0 and b1 in self.robot_bodies:
                touching.add(int(b1))
        return touching

    def head_top_z(self):
        R = self.data.xmat[self.head].reshape(3, 3)
        return float((R @ HEAD_TOP_AXIS)[2])

    def head_heading(self):
        """Yaw of the head z-axis on the floor: the headstand command frame
        (same as headstand_mdp.head_heading)."""
        R = self.data.xmat[self.head].reshape(3, 3)
        return math.atan2(R[1, 2], R[0, 2])

    def state(self):
        """Per-step record used by the judges (see scenarios.py)."""
        d = self.data
        R = d.xmat[self.trunk].reshape(3, 3)
        touching = self.floor_contacts()
        head_top_down = self.head_top_z() < -TOP_DOWN_MIN
        return {
            "t": self.t,
            "x": float(d.qpos[self.free_q]),
            "y": float(d.qpos[self.free_q + 1]),
            "z": float(d.qpos[self.free_q + 2]),
            "upright": float(R[2, 2]),        # trunk z-axis . world z: 1 upright, 0 lying
            "pitch_rate": float(d.qvel[self.free_v + 4]),  # body frame, + = forward roll
            "trunk_yaw": float(math.atan2(R[1, 0], R[0, 0])),
            # trunk angular velocity in the WORLD frame (free-joint qvel is body frame)
            **dict(zip(("wx", "wy", "wz"), (R @ d.qvel[self.free_v + 3:self.free_v + 6]).tolist())),
            "head_x": float(d.xpos[self.head][0]),
            "head_y": float(d.xpos[self.head][1]),
            "com_x": float(d.subtree_com[self.trunk][0]),   # whole-robot COM (trunk is the root)
            "com_y": float(d.subtree_com[self.trunk][1]),
            "heading": self.head_heading(),
            "on_head": bool(head_top_down and touching == {self.head}),
            "non_foot_contact": bool(touching - self.feet),
        }

    # ---- rendering ---------------------------------------------------------------------

    def make_camera(self, distance=0.9, elevation=-12.0, azimuth=135.0):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth
        return cam

    def render(self, renderer, cam):
        cam.lookat[:] = self.data.qpos[self.free_q:self.free_q + 3]
        renderer.update_scene(self.data, camera=cam)
        return renderer.render()
