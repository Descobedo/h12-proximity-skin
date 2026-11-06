"""Simulate one skinned H1-2 arm at work: the skin and the lidar sensing, the arm reacting.

    python3 -m golem_proximity_skin.simulate --arm right --video skin_sim.mp4
    python3 -m golem_proximity_skin.simulate --arm left --viewer      # live, interactive

Standalone MuJoCo, no ROS or containers. The H1-2 stands with its pelvis welded
in place (CL_Assets' scene_h1_2_magpie_pelvis_fixed.xml) at a table. Only the
chosen arm carries skin (the layout's skins whose parent is on that side).

What runs, each on its own clock in sim time:
  physics      500 Hz: PD + gravity compensation on every body joint (the
               model's motors are torque actuators); the arm tracks a looping
               reach-and-sweep task, the rest holds its pose.
  skin         at the fitted model's rate (12.4 Hz, HIRO's firmware): every
               sensor's 8x8 zones ray-cast (mujoco_tof, the caster the RoboCasa sim
               uses), with the noise and invalid-zone flicker fitted from real
               recordings (sensor_model.py; --ideal-sensor for exact distances).
  lidar        10 Hz: a MID-360-like scan from livox_link, as golem_robocasa's
               h12_sensors casts it (coarser grid), robot returns dropped.
  reaction     on each skin cycle: the arm's speed along its task scales from
               1 (nearest obstacle >= --slow-at) to 0 (<= --stop-at), so it
               slows, then holds, while something is close, and resumes after.

A person walks up to the table (the lidar sees them first), reaches into the
arm's workspace (the skin sees the hand and forearm), and walks away. Returns
from the robot's own body are told apart by the geom they hit, which the sim
knows and the hardware does not: they are drawn grey and do not slow the arm.

Outputs: a video (--video, .mp4 needs imageio[ffmpeg], else .gif), a plot of
distances and arm speed (--plot), and a recording (--record .npz) of every
skin cycle: readings, the geoms hit, arm joint angles, speed, distances.
"""
import argparse
import math
import os
import time

import mujoco
import numpy as np

from . import vl53l5cx
from .layout import load_layout, select_parents
from .mujoco_tof import ToFSkinCaster, _multi_ray
from .sensor_model import DEFAULT_MODEL, SensorModel
from .preview import add_dot, add_fov, build_spec, sensor_hues

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LAYOUT = os.path.join(HERE, '..', 'config', 'h12_skin.yaml')
DEFAULT_MJCF = os.path.join(HERE, '..', '..', '..', '..', 'CL_Assets', 'mujoco_assets',
                            'scene_h1_2_magpie_pelvis_fixed.xml')

ARM_JOINTS = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
              'wrist_roll', 'wrist_pitch', 'wrist_yaw']

# The task, right arm, joint space (rad, ARM_JOINTS order): a loop of
# ready -> reach over the table -> sweep outward -> low over the bin -> ready.
# The left arm mirrors roll, yaw and wrist roll/yaw.
WAYPOINTS = np.array([
    [-0.30, -0.20, 0.00, 0.70, 0.0, 0.0, 0.0],
    [-1.00, -0.10, 0.20, 0.25, 0.0, 0.2, 0.0],
    [-0.85, -0.65, 0.35, 0.45, 0.4, 0.0, 0.0],
    [-0.55, -0.30, 0.00, 0.95, 0.0, -0.2, 0.0],
])
SEGMENT_S = 2.5            # s per waypoint segment at full speed
MIRROR = np.array([1, -1, -1, 1, -1, 1, -1])

# PD gains (N*m/rad, N*m*s/rad), from the MJCF's position classes.
ARM_KP, ARM_KD = 120.0, 3.0
BODY_KP, BODY_KD = 300.0, 6.0

LIDAR_AZ, LIDAR_EL, LIDAR_HZ, LIDAR_RANGE = 120, 24, 10.0, 40.0

SKIN_RGBA = np.array([0.25, 0.7, 0.9, 0.35])
NEAR_RGBA = np.array([1.0, 0.15, 0.1, 0.6])


# -- the person -------------------------------------------------------------
REACH_T = 5.0                  # s, the person starts reaching
FINGERTIP_GAP = 0.06           # m, the reach stops this far from the arm's skin
PERSON_FOREARM_HALF = 0.2      # m, half-length of the person's forearm capsule
PERSON_ARM_RADIUS = 0.045      # m, its radius


# -- scene -------------------------------------------------------------------

def add_world(spec, side_sign: float):
    """Table, bin and shelf on the arm's side; a mocap 'person' (visual only, ray-cast)."""
    y = 0.28 * side_sign
    wb = spec.worldbody
    G = mujoco.mjtGeom
    wb.add_geom(name='table_top', type=G.mjGEOM_BOX, size=[0.25, 0.32, 0.02],
                pos=[0.58, y, 0.70], rgba=[0.62, 0.47, 0.32, 1])
    wb.add_geom(name='table_base', type=G.mjGEOM_BOX, size=[0.08, 0.08, 0.34],
                pos=[0.58, y, 0.34], rgba=[0.3, 0.3, 0.3, 1])
    wb.add_geom(name='bin', type=G.mjGEOM_BOX, size=[0.07, 0.09, 0.06],
                pos=[0.52, y + 0.08 * side_sign, 0.78], rgba=[0.2, 0.5, 0.8, 1])
    wb.add_geom(name='shelf', type=G.mjGEOM_BOX, size=[0.3, 0.02, 0.5],
                pos=[0.35, 0.95 * side_sign, 1.1], rgba=[0.75, 0.75, 0.7, 1])
    person = wb.add_body(name='person', mocap=True, pos=[3.0, y, 0.0])
    skin_tone = [0.85, 0.65, 0.5, 1]
    shirt = [0.35, 0.3, 0.55, 1]
    vis = dict(contype=0, conaffinity=0)
    person.add_geom(name='person_legs', type=G.mjGEOM_CAPSULE, size=[0.09, 0.42],
                    pos=[0, 0, 0.47], rgba=[0.2, 0.2, 0.25, 1], **vis)
    person.add_geom(name='person_torso', type=G.mjGEOM_CAPSULE, size=[0.16, 0.25],
                    pos=[0, 0, 1.2], rgba=shirt, **vis)
    person.add_geom(name='person_head', type=G.mjGEOM_SPHERE, size=[0.11, 0, 0],
                    pos=[0, 0, 1.68], rgba=skin_tone, **vis)
    # The reaching forearm, along x (toward the robot); a mocap body of its own
    # (MuJoCo allows mocap bodies only as children of the world).
    arm = wb.add_body(name='person_arm', mocap=True, pos=[3.0, y, 1.3])
    arm.add_geom(name='person_forearm', type=G.mjGEOM_CAPSULE, size=[PERSON_ARM_RADIUS, PERSON_FOREARM_HALF],
                 pos=[0, 0, 0], quat=[0.7071068, 0, 0.7071068, 0], rgba=skin_tone, **vis)


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def person_pose(t, table_y, reach_to):
    """(body xyz, forearm centre xyz) of the person at sim time t (s); the forearm's
    centre goes to reach_to (world) while reaching."""
    walk_in, reach, hold, withdraw, walk_out = 1.0, REACH_T, 6.2, 8.0, 9.5
    far, near = 3.0, 1.15
    if t < reach:
        x = far + (near - far) * _smooth((t - walk_in) / (reach - 0.4 - walk_in))
    elif t < walk_out:
        x = near
    else:
        x = near + (far - near) * _smooth((t - walk_out) / 3.0)
    ext = _smooth((t - reach) / (hold - reach)) * (1.0 - _smooth((t - withdraw) / 1.0))
    body = np.array([x, table_y, 0.0])
    rest = body + [-0.05, 0.0, 1.05]
    return body, rest + (np.asarray(reach_to) - rest) * ext


# -- the arm's controller -----------------------------------------------------

class ArmTask:
    """Phase along the looping waypoint path; advanced at the commanded speed scale."""

    def __init__(self, waypoints):
        self.wp = waypoints
        self.phase = 0.0          # in segments

    def advance(self, dt, scale):
        self.phase = (self.phase + dt * scale / SEGMENT_S) % len(self.wp)

    def target(self):
        i = int(self.phase)
        u = _smooth(self.phase - i)
        return self.wp[i] + (self.wp[(i + 1) % len(self.wp)] - self.wp[i]) * u


class Controller:
    """PD + gravity compensation on every hinge joint with a motor."""

    def __init__(self, model, data, arm_side):
        self.model = model
        self.act, self.qadr, self.dadr, self.kp, self.kd = [], [], [], [], []
        self.arm_idx = {}
        for a in range(model.nu):
            if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            j = model.actuator_trnid[a, 0]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
            if model.actuator_gainprm[a, 0] != 1.0 or 'hinge' in name:
                continue      # gripper fingers: their own position actuators
            arm = next((k for k, jn in enumerate(ARM_JOINTS) if name == f'{arm_side}_{jn}_joint'), None)
            is_arm = any(p in name for p in ('shoulder', 'elbow', 'wrist'))
            if arm is not None:
                self.arm_idx[arm] = len(self.act)
            self.act.append(a)
            self.qadr.append(model.jnt_qposadr[j])
            self.dadr.append(model.jnt_dofadr[j])
            self.kp.append(ARM_KP if is_arm else BODY_KP)
            self.kd.append(ARM_KD if is_arm else BODY_KD)
        self.act, self.qadr, self.dadr = (np.array(v) for v in (self.act, self.qadr, self.dadr))
        self.kp, self.kd = np.array(self.kp), np.array(self.kd)
        self.q_ref = data.qpos[self.qadr].copy()
        self.lo = model.actuator_forcerange[self.act, 0]
        self.hi = model.actuator_forcerange[self.act, 1]
        jids = model.actuator_trnid[self.act, 0]
        self.jlo, self.jhi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
        self.limited = model.jnt_limited[jids].astype(bool)

    def set_arm(self, q_arm):
        for k, idx in self.arm_idx.items():
            q = q_arm[k]
            if self.limited[idx]:
                q = min(max(q, self.jlo[idx]), self.jhi[idx])
            self.q_ref[idx] = q

    def arm_q(self, data):
        return np.array([data.qpos[self.qadr[self.arm_idx[k]]] for k in range(len(ARM_JOINTS))])

    def apply(self, data):
        q, qd = data.qpos[self.qadr], data.qvel[self.dadr]
        tau = self.kp * (self.q_ref - q) - self.kd * qd + data.qfrc_bias[self.dadr]
        lim = self.hi > self.lo
        tau = np.where(lim, np.clip(tau, self.lo, self.hi), tau)
        data.ctrl[self.act] = tau


# -- lidar --------------------------------------------------------------------

class Lidar:
    """MID-360-like scan from livox_link: golem_robocasa/h12_sensors.py's ray pattern on
    a coarser grid (local elevation -52..+7 deg: the sensor is mounted upside down)."""

    def __init__(self, model, robot_geom):
        self.model = model
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'livox_link')
        self.torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
        if self.body < 0 or self.torso < 0:
            raise KeyError('livox_link / torso_link not in the model')
        az = np.linspace(0, 2 * np.pi, LIDAR_AZ, endpoint=False)
        el = np.linspace(-np.radians(52), np.radians(7), LIDAR_EL, endpoint=False)
        A, E = np.meshgrid(az, el)
        self.local = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        self.robot_geom = robot_geom
        n = len(self.local)
        self._d = np.full((n, 1), -1.0)
        self._g = np.full((n, 1), -1, dtype=np.int32)

    def scan(self, data):
        """(n, 3) world points of the scan's returns off anything but the robot."""
        o = data.xpos[self.body].copy()
        R = data.xmat[self.body].reshape(3, 3)
        dirs = self.local @ R.T
        self._d.fill(-1.0)
        self._g.fill(-1)
        _multi_ray(self.model, data, o.reshape(3, 1), np.ascontiguousarray(dirs).reshape(-1, 1),
                   self._g, self._d, self.torso, LIDAR_RANGE)
        d, g = self._d.ravel(), self._g.ravel()
        keep = (d >= 0.1) & (g >= 0) & ~self.robot_geom[np.where(g >= 0, g, 0)]
        return o + dirs[keep] * d[keep, None]


# -- rendering ----------------------------------------------------------------

def proximity_rgba(d, near, far, alpha=0.9):
    if not math.isfinite(d):
        return [0.6, 0.6, 0.6, 0.35]
    t = min(max((d - near) / (far - near), 0.0), 1.0)
    return [min(1.0, 2.0 * (1.0 - t)), min(1.0, 2.0 * t), 0.1, alpha]


class Overlay:
    def __init__(self, caster, args):
        self.caster = caster
        self.args = args
        n = sum(len(s.sensors) for s in caster.skins)
        self.hues = sensor_hues(n)

    def draw(self, scene, data, frame):
        a = self.args
        if frame.lidar is not None and a.lidar:
            pts = frame.lidar
            if len(pts) > 1500:
                pts = pts[np.linspace(0, len(pts) - 1, 1500).astype(int)]
            for p in pts:
                add_dot(scene, p, 0.008, [1.0, 1.0, 1.0, 0.7])
        k = 0
        for skin, poses in zip(self.caster.skins, self.caster.sensor_world_poses(data)):
            r_mm, gids = frame.readings[skin.name], frame.geoms[skin.name]
            for i, (origin, R) in enumerate(poses):
                hue = self.hues[k]
                k += 1
                env = (r_mm[i] != vl53l5cx.NO_TARGET) & ~frame.self_hit(gids[i])
                own = (r_mm[i] != vl53l5cx.NO_TARGET) & frame.self_hit(gids[i])
                for z in np.nonzero(env)[0]:
                    add_dot(scene, origin + R @ (vl53l5cx.ZONE_RAYS[z] * r_mm[i, z] / 1000.0), 0.007, [*hue, 1.0])
                for z in np.nonzero(own)[0][::3]:
                    add_dot(scene, origin + R @ (vl53l5cx.ZONE_RAYS[z] * r_mm[i, z] / 1000.0), 0.004,
                            [0.5, 0.5, 0.5, 0.6])
                d = r_mm[i][env].min() / 1000.0 if env.any() else math.inf
                depth = min(d, a.fov_draw) if math.isfinite(d) else 0.06
                add_fov(scene, origin, R, depth, proximity_rgba(d, a.stop_at, a.slow_at), 1.5)


class Frame:
    """One skin cycle's results."""

    def __init__(self, readings, geoms, robot_geom, lidar):
        self.readings, self.geoms, self.lidar = readings, geoms, lidar
        self._robot = robot_geom

    def self_hit(self, gids):
        return (gids >= 0) & self._robot[np.where(gids >= 0, gids, 0)]

    def env_nearest(self):
        best = math.inf
        for name, r in self.readings.items():
            env = (r != vl53l5cx.NO_TARGET) & ~self.self_hit(self.geoms[name])
            if env.any():
                best = min(best, r[env].min() / 1000.0)
        return best


def banner(img, lines):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, 'RGBA')
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
    except OSError:
        try:
            font = ImageFont.truetype('arial.ttf', 18)
        except OSError:
            font = ImageFont.load_default()
    h = 26 * len(lines) + 12
    d.rectangle([0, 0, im.width, h], fill=(0, 0, 0, 150))
    for i, (text, color) in enumerate(lines):
        d.text((12, 8 + 26 * i), text, fill=color, font=font)
    return np.asarray(im)


class VideoWriter:
    def __init__(self, path, fps):
        self.path, self.fps, self.frames, self._w = path, fps, [], None
        if path.lower().endswith('.mp4'):
            try:
                import imageio.v2 as imageio
                self._w = imageio.get_writer(path, fps=fps, codec='libx264', quality=7,
                                             macro_block_size=8)
            except Exception as e:     # imageio or its ffmpeg missing
                self.path = os.path.splitext(path)[0] + '.gif'
                print(f'mp4 unavailable ({e.__class__.__name__}); writing {self.path}')

    def add(self, img):
        if self._w is not None:
            self._w.append_data(img)
        else:
            self.frames.append(img)

    def close(self):
        if self._w is not None:
            self._w.close()
            return
        from PIL import Image
        ims = [Image.fromarray(f).convert('P', palette=Image.ADAPTIVE, colors=192) for f in self.frames]
        ims[0].save(self.path, save_all=True, append_images=ims[1:], duration=int(1000 / self.fps), loop=0)


# -- the run --------------------------------------------------------------------

def run(args):
    side = args.arm
    sign = -1.0 if side == 'right' else 1.0
    layout = select_parents(load_layout(args.layout), f'{side}_')
    layout.skins = [s for s in layout.skins if any(k in s.parent for k in ('shoulder', 'elbow', 'wrist'))]
    if not layout.skins:
        raise SystemExit(f'no {side} arm skin in {args.layout}')
    spec = build_spec(args.mjcf, layout)
    add_world(spec, sign)
    model = spec.compile()
    data = mujoco.MjData(model)

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
    robot_geom = model.body_rootid[model.geom_bodyid] == model.body_rootid[pelvis]
    dermis = {s.name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{s.name}_dermis')
              for s in layout.skins}
    person = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'person')]
    person_arm = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'person_arm')]

    waypoints = WAYPOINTS if side == 'right' else WAYPOINTS * MIRROR
    task = ArmTask(waypoints)
    ctrl = Controller(model, data, side)
    ctrl.set_arm(task.target())
    data.qpos[ctrl.qadr] = ctrl.q_ref
    mujoco.mj_forward(model, data)

    sensor = None if args.ideal_sensor else SensorModel.load(args.sensor_model)
    if sensor is not None:
        layout.rate_hz = sensor.rate_hz
    caster = ToFSkinCaster(model, layout, sensor_model=sensor, seed=args.seed)
    lidar = Lidar(model, robot_geom)
    overlay = Overlay(caster, args)
    print(f'{side} arm: {len(layout.skins)} skins, {layout.sensor_count()} VL53L5CX sensors at '
          f'{layout.rate_hz:.1f} Hz ({"ideal" if sensor is None else "fitted sensor model"}); '
          f'lidar {LIDAR_AZ}x{LIDAR_EL} rays at {LIDAR_HZ:g} Hz; {args.duration:g} s')

    dt = model.opt.timestep
    skin_period, lidar_period = 1.0 / layout.rate_hz, 1.0 / LIDAR_HZ
    next_skin, next_lidar = 0.0, 0.0
    scale, scale_cmd = 1.0, 1.0
    lidar_pts = None
    frame = None
    log = {k: [] for k in ('t', 'scale', 'skin_nearest', 'lidar_clearance', 'q_arm', 'phase')}
    rec = {s.name: [] for s in layout.skins}
    rec_g = {s.name: [] for s in layout.skins}

    renderer = writer = viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
    elif args.video:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
        renderer = mujoco.Renderer(model, args.height, args.width, max_geom=15000)
        writer = VideoWriter(args.video, layout.rate_hz)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.35, 0.25 * sign, 1.05]
        cam.azimuth = 245.0 if side == 'right' else 115.0
        cam.elevation, cam.distance = -20.0, 2.5

    # The person reaches in alongside the outer side of the forearm skin (-y for the
    # right arm), where its sensors look, re-aiming every skin cycle as the arm moves:
    # their forearm (along x) FINGERTIP_GAP from the skin. HIRO's forearm sensors face
    # out and up, none along the forearm, so a hand straight ahead of it is unseen.
    aim_skins = [k for k, s in enumerate(caster.skins) if s.name.endswith('forearm')] or         list(range(len(caster.skins)))
    out_y = sign * 1.0             # the arm's outer side: +y for the left, -y for the right

    def aim():
        poses = caster.sensor_world_poses(data)
        pts = np.array([o for k in aim_skins for o, _ in poses[k]])
        side = pts[:, 1].max() if out_y > 0 else pts[:, 1].min()
        return np.array([pts[:, 0].mean() + PERSON_FOREARM_HALF - 0.05,
                         side + out_y * (FINGERTIP_GAP + PERSON_ARM_RADIUS),
                         pts[:, 2].mean()])

    reach_to = aim()
    wall0 = time.monotonic()
    while data.time < args.duration:
        t = data.time
        body, hand = person_pose(t, 0.28 * sign, reach_to)
        data.mocap_pos[person] = body
        data.mocap_pos[person_arm] = hand

        if t >= next_lidar and args.lidar:
            next_lidar += lidar_period
            lidar_pts = lidar.scan(data)

        if t >= next_skin:
            next_skin += skin_period
            reach_to = aim()
            readings, geoms = caster.cast(data, with_geoms=True)
            frame = Frame(readings, geoms, robot_geom, lidar_pts)
            d = frame.env_nearest()
            scale_cmd = 1.0 if not args.react else float(
                np.clip((d - args.stop_at) / (args.slow_at - args.stop_at), 0.0, 1.0))
            arm_pts = np.array([p[0] for poses in caster.sensor_world_poses(data) for p in poses])
            if lidar_pts is not None and len(lidar_pts):
                diff = lidar_pts[:, None, :] - arm_pts[None, :, :]
                clearance = float(np.sqrt((diff ** 2).sum(-1)).min())
            else:
                clearance = math.inf
            for s in layout.skins:
                rec[s.name].append(readings[s.name])
                rec_g[s.name].append(geoms[s.name])
                ds = readings[s.name][~frame.self_hit(geoms[s.name]) & (readings[s.name] != 0)]
                near = ds.min() / 1000.0 if ds.size else math.inf
                if dermis[s.name] >= 0:
                    w = 1.0 - min(max((near - args.stop_at) / (args.slow_at - args.stop_at), 0.0), 1.0)
                    model.geom_rgba[dermis[s.name]] = SKIN_RGBA + (NEAR_RGBA - SKIN_RGBA) * w
            for k, v in (('t', t), ('scale', scale), ('skin_nearest', d), ('lidar_clearance', clearance),
                         ('q_arm', ctrl.arm_q(data)), ('phase', task.phase)):
                log[k].append(v)

            state = 'STOPPED' if scale < 0.02 else 'SLOWING' if scale < 0.98 else 'MOVING'
            color = {'STOPPED': (255, 90, 70), 'SLOWING': (255, 210, 60), 'MOVING': (120, 230, 120)}[state]
            text = [
                (f't = {t:5.2f} s   {side} arm {state} ({100 * scale:3.0f}% speed)', color),
                (f'skin nearest obstacle: {d:5.3f} m' if math.isfinite(d) else 'skin nearest obstacle:  none',
                 (255, 255, 255)),
                (f'lidar clearance to arm: {clearance:5.2f} m' if math.isfinite(clearance)
                 else 'lidar clearance to arm:  none', (220, 220, 220)),
            ]
            if renderer is not None:
                renderer.update_scene(data, cam)
                overlay.draw(renderer.scene, data, frame)
                writer.add(banner(renderer.render(), text))
            if viewer is not None:
                if not viewer.is_running():
                    break
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    overlay.draw(viewer.user_scn, data, frame)
                viewer.sync()
                lag = data.time - (time.monotonic() - wall0)
                if lag > 0:
                    time.sleep(lag)
            if len(log['t']) % int(layout.rate_hz) == 0:
                print('  ' + ' | '.join(s for s, _ in text))

        # First-order filter (0.15 s) on the commanded speed: no step changes in arm velocity.
        scale += (scale_cmd - scale) * min(1.0, dt / 0.15)
        task.advance(dt, scale)
        ctrl.set_arm(task.target())
        ctrl.apply(data)
        mujoco.mj_step(model, data)

    if writer is not None:
        writer.close()
        print(f'wrote {writer.path}')
    if viewer is not None:
        # Keep the window up until the user closes it: the viewer's render thread
        # must finish before the process exits, or glfw is torn down under it.
        if viewer.is_running():
            print('run finished: close the viewer window to exit')
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
        time.sleep(0.5)

    out = {k: np.array(v) for k, v in log.items()}
    if args.record:
        np.savez_compressed(
            args.record, timestamps=out['t'], arm=side, arm_joints=np.array(ARM_JOINTS),
            q_arm=out['q_arm'], speed_scale=out['scale'], skin_nearest=out['skin_nearest'],
            lidar_clearance=out['lidar_clearance'],
            **{f'tof_{n}': np.array(v) for n, v in rec.items()},
            **{f'self_hit_{n}': np.array([[robot_geom[np.where(g >= 0, g, 0)] & (g >= 0)
                                           for g in step] for step in v]) for n, v in rec_g.items()})
        print(f'wrote {args.record}')
    if args.plot:
        plot(out, args, side)
    summarize(out, args)
    return out


def plot(out, args, side):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib missing: no plot')
        return
    t = out['t']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    skin = np.where(np.isfinite(out['skin_nearest']), out['skin_nearest'], np.nan)
    lid = np.where(np.isfinite(out['lidar_clearance']), out['lidar_clearance'], np.nan)
    ax1.plot(t, lid, color='0.45', label='lidar: nearest point to the arm')
    ax1.plot(t, skin, color='tab:blue', lw=2, label='skin: nearest return (robot excluded)')
    ax1.axhline(args.slow_at, color='tab:orange', ls='--', lw=1, label=f'slow at {args.slow_at:g} m')
    ax1.axhline(args.stop_at, color='tab:red', ls='--', lw=1, label=f'stop at {args.stop_at:g} m')
    ax1.set_ylim(0, 2.0)
    ax1.set_ylabel('distance (m)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title(f'H1-2 {side} arm with proximity skin')
    ax2.plot(t, 100 * out['scale'], color='tab:green', lw=2)
    ax2.set_ylabel('arm speed (%)')
    ax2.set_xlabel('sim time (s)')
    ax2.set_ylim(-5, 105)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120)
    print(f'wrote {args.plot}')


def summarize(out, args):
    s, t = out['scale'], out['t']
    stopped = t[s < 0.02]
    d = out['skin_nearest']
    print(f"skin cycles: {len(t)}; nearest obstacle seen by the skin: {np.nanmin(np.where(np.isfinite(d), d, np.nan)):.3f} m")
    if stopped.size:
        after = t[t > stopped[-1]]
        end = f'resumed by t = {after[0]:.2f} s' if after.size else 'still held when the run ended'
        print(f'arm held (speed < 2%) for {stopped.size / (len(t) / args.duration):.1f} s, '
              f'first at t = {stopped[0]:.2f} s, {end}')
    else:
        print('arm never fully stopped')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm', choices=('right', 'left'), default='right')
    ap.add_argument('--mjcf', default=DEFAULT_MJCF, help='fixed-pelvis H1-2 scene MJCF')
    ap.add_argument('--layout', default=DEFAULT_LAYOUT)
    ap.add_argument('--duration', type=float, default=14.0, help='sim seconds')
    ap.add_argument('--video', default='skin_sim.mp4', help='.mp4 or .gif; "" for none')
    ap.add_argument('--plot', default='skin_sim.png', help='"" for none')
    ap.add_argument('--record', default='skin_sim.npz', help='"" for none')
    ap.add_argument('--viewer', action='store_true', help='interactive viewer, real time (no video)')
    ap.add_argument('--no-react', dest='react', action='store_false', help='arm ignores the skin')
    ap.add_argument('--no-lidar', dest='lidar', action='store_false')
    ap.add_argument('--ideal-sensor', action='store_true', help='no noise or flicker (first-hit distances)')
    ap.add_argument('--sensor-model', default=DEFAULT_MODEL, help='fitted VL53L5CX model (sensor_model.py)')
    ap.add_argument('--seed', type=int, default=0, help='sensor-noise seed')
    ap.add_argument('--slow-at', type=float, default=0.25, help='m: skin range where the arm starts slowing')
    ap.add_argument('--stop-at', type=float, default=0.10, help='m: skin range where the arm holds')
    ap.add_argument('--fov-draw', type=float, default=0.3, help='m, longest FOV edge drawn')
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=720)
    args = ap.parse_args(argv)
    if args.slow_at <= args.stop_at:
        ap.error('--slow-at must be above --stop-at')
    run(args)


if __name__ == '__main__':
    main()
