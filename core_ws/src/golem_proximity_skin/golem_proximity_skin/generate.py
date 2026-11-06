"""Generate proximity-skin training data: randomised, seeded episodes, headless, in parallel.

    python3 -m golem_proximity_skin.generate --out data/arm_v1 --skins right_shoulder \
        right_upper_arm right_forearm --episodes 1000 --workers 8
    python3 -m golem_proximity_skin.generate --out data/knee_v1 --skins right_knee --episodes 500

Each episode moves the limbs that carry the chosen skins along a random smooth
joint trajectory (kinematic playback; the pelvis is fixed) in a random scene, and
casts every sensor at the fitted sensor model's rate. Episode types, after ARMOR
(arXiv 2412.00396):
  avoid     static obstacles placed tight around the limb's path, in its sensors'
            view, never touching it (clearance >= --min-clearance at every frame)
  dynamic   static clutter, plus one obstacle (a hand, forearm or object) moving
            in toward the limb; it may make contact
  free      no obstacles
Per episode, also randomised: an optional table and wall, each sensor's mounting
(+-2 mm, +-2 deg), and the odd dead sensor (its readings all no-target). Zones the
real skin obscures (the board map's masked_zones, hiro_udp) read no-target and raw 0
in every episode, as hiro_udp reports them from the hardware.

What a model may see (the hardware has it too), per frame and sensor:
  readings_mm   (T, S, 64) uint16   sensor model output, 0 = no target / invalid
  raw_mm        (T, S, 64) uint16   the firmware's raw distance, invalid zones too
  q, qd         (T, J)              the moved joints (rad, rad/s), joint_names
Labels (the sim knows, the hardware does not):
  ideal_mm      (T, S, 64) uint16   exact first-hit distance, 0 = no hit in range
  hit_class     (T, S, 64) uint8    0 none, 1 robot (self), 2 static obstacle,
                                    3 moving obstacle, 4 world (floor, table, wall)
  hit_obstacle  (T, S, 64) int8     obstacle index hit, -1 none
  nearest_env_mm (T, S)    uint16   nearest non-robot ideal return, 0 = none
  clearance_m   (T, K)     float32  per skin: min distance from its link (and skin)
                                    to any obstacle (< 0: penetrating), capped
  sensor_pos / sensor_quat (T, S, 3 / 4) world pose of each sensor (quat wxyz)
  obstacles_*   (T, O, ...)         type, size, position, quaternion, velocity
Plus q_cmd (= q here: the trajectory), sensor_alive (S,), zone_masked (S, 64), mount_offset (S, 6),
episode type and seed, and skin/sensor names. One .npz per episode in --out, a
manifest.jsonl line per episode, and dataset.json with the configuration,
layout and sensor model used. dataset.py loads them.

Limits: motion is random, not an expert avoiding obstacles (for policy learning
an expert, e.g. retargeted human motion plus a planner, is still needed); all
sensors of a frame are cast at one instant (the firmware reads a board's sensors
one after another); the sensor model has no bias term and one ray per zone.
"""
import argparse
import json
import math
import os
import time
from multiprocessing import get_context

import mujoco
import numpy as np

from . import vl53l5cx
from .layout import load_layout
from .mujoco_tof import ToFSkinCaster, _multi_ray
from .preview import build_spec
from .sensor_model import DEFAULT_MODEL, SensorModel

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LAYOUT = os.path.join(HERE, '..', 'config', 'h12_skin.yaml')
DEFAULT_BOARDS = os.path.join(HERE, '..', 'config', 'hiro_boards.yaml')
DEFAULT_MJCF = os.path.join(HERE, '..', '..', '..', '..', 'CL_Assets', 'mujoco_assets',
                            'scene_h1_2_magpie_pelvis_fixed.xml')
SCHEMA = 1

# The joints each limb moves, and the range (rad) sampled for each: inside the
# joint limits and clear of the gross self-collisions (arm into torso, legs crossing).
# Left limbs mirror roll and yaw.
ARM = [('shoulder_pitch', -1.5, 0.4), ('shoulder_roll', -1.0, 0.1), ('shoulder_yaw', -0.7, 0.7),
       ('elbow', -0.3, 1.5), ('wrist_roll', -1.2, 1.2), ('wrist_pitch', -0.4, 0.4),
       ('wrist_yaw', -0.9, 0.9)]
LEG = [('hip_yaw', -0.3, 0.3), ('hip_pitch', -1.0, 0.4), ('hip_roll', -0.35, 0.05),
       ('knee', 0.0, 1.6), ('ankle_pitch', -0.6, 0.4), ('ankle_roll', -0.2, 0.2)]
MIRRORED = ('roll', 'yaw')
ARM_LINKS = ('shoulder', 'elbow', 'wrist')
LEG_LINKS = ('hip', 'knee', 'ankle')

POOL = {'box': 8, 'sphere': 4, 'cylinder': 4, 'capsule': 4}   # obstacle geoms, reused per episode
TYPE_IDS = {'box': 0, 'sphere': 1, 'cylinder': 2, 'capsule': 3}
GEOM_TYPES = {'box': mujoco.mjtGeom.mjGEOM_BOX, 'sphere': mujoco.mjtGeom.mjGEOM_SPHERE,
              'cylinder': mujoco.mjtGeom.mjGEOM_CYLINDER, 'capsule': mujoco.mjtGeom.mjGEOM_CAPSULE}
PARK = np.array([0.0, 0.0, -20.0])     # where unused obstacles wait, out of every sensor's range
HIT_NONE, HIT_SELF, HIT_STATIC, HIT_MOVING, HIT_WORLD = 0, 1, 2, 3, 4
CLEARANCE_CAP = 1.0                    # m: clearances beyond this are stored as this


# -- the model -----------------------------------------------------------------

def _limb(parent):
    side = 'right' if parent.startswith('right_') else 'left'
    if any(k in parent for k in ARM_LINKS):
        return side, ARM
    if any(k in parent for k in LEG_LINKS):
        return side, LEG
    raise ValueError(f'no limb for skin parent {parent!r}')


class World:
    """The H1-2 with the chosen skins, an obstacle pool, a table and a wall: built once
    per worker, re-dressed per episode."""

    def __init__(self, mjcf, layout_path, skins, sensor_model_path, boards_path=None):
        layout = load_layout(layout_path)
        byname = {s.name: s for s in layout.skins}
        missing = [n for n in skins if n not in byname]
        if missing:
            raise ValueError(f'skins not in the layout: {missing}')
        layout.skins = [byname[n] for n in skins]
        for s in layout.skins:
            s.enabled = True
        self.sensor = SensorModel.load(sensor_model_path) if sensor_model_path else None
        layout.rate_hz = self.sensor.rate_hz if self.sensor else layout.rate_hz
        self.layout = layout

        spec = build_spec(mjcf, layout)
        self.pool = []                       # (kind, body name, geom name)
        for kind, n in POOL.items():
            for i in range(n):
                b = spec.worldbody.add_body(name=f'obs_{kind}_{i}', mocap=True, pos=PARK.tolist())
                b.add_geom(name=f'obs_{kind}_{i}_g', type=GEOM_TYPES[kind], size=[0.05, 0.05, 0.05],
                           rgba=[0.8, 0.5, 0.2, 1], contype=0, conaffinity=0)
                self.pool.append((kind, b.name, f'obs_{kind}_{i}_g'))
        for name in ('table', 'wall'):
            b = spec.worldbody.add_body(name=name, mocap=True, pos=PARK.tolist())
            b.add_geom(name=f'{name}_g', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.3, 0.02],
                       rgba=[0.6, 0.5, 0.4, 1], contype=0, conaffinity=0)
        self.model = spec.compile()
        m = self.model
        self.data = mujoco.MjData(m)
        mujoco.mj_forward(m, self.data)

        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.obs_geom = np.array([gid(g) for _, _, g in self.pool])
        self.obs_mocap = np.array([m.body_mocapid[bid(b)] for _, b, _ in self.pool])
        self.obs_kind = [k for k, _, _ in self.pool]
        self.extra = {n: (m.body_mocapid[bid(n)], gid(f'{n}_g')) for n in ('table', 'wall')}

        pelvis = bid('pelvis')
        self.robot_geom = m.body_rootid[m.geom_bodyid] == m.body_rootid[pelvis]
        self.geom_class = np.full(m.ngeom, HIT_WORLD, dtype=np.uint8)
        self.geom_class[self.robot_geom] = HIT_SELF
        self.geom_obstacle = np.full(m.ngeom, -1, dtype=np.int8)
        self.geom_obstacle[self.obs_geom] = np.arange(len(self.obs_geom))

        # The moved joints: every limb that carries a chosen skin.
        self.joint_names, lo, hi = [], [], []
        for side in sorted({_limb(s.parent)[0] + ':' + ('arm' if _limb(s.parent)[1] is ARM else 'leg')
                            for s in layout.skins}):
            sd, kind = side.split(':')
            for jn, a, b in (ARM if kind == 'arm' else LEG):
                if sd == 'left' and any(k in jn for k in MIRRORED):
                    a, b = -b, -a
                name = f'{sd}_{jn}_joint'
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                jl, jh = m.jnt_range[j]
                self.joint_names.append(name)
                lo.append(max(a, jl))
                hi.append(min(b, jh))
        self.qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                              for n in self.joint_names])
        self.q_lo, self.q_hi = np.array(lo), np.array(hi)

        # Each skin's surface for clearance: its parent link's geoms and its skin mesh.
        self.skin_geoms = []
        for s in layout.skins:
            b = bid(s.parent)
            self.skin_geoms.append(np.nonzero(m.geom_bodyid == b)[0])

        self.caster = ToFSkinCaster(m, layout, sensor_model=self.sensor)
        self.base_sensors = [list(s) for s in self.caster._sensors]
        self.sensor_names = [f'{s.name}_sensor_{i}' for s in layout.skins for i in range(len(s.sensors))]
        self.n_sensors = len(self.sensor_names)
        # Zones the real skin obscures, per layout sensor, from the board map (hiro_udp).
        self.zone_masked = np.zeros((self.n_sensors, vl53l5cx.NUM_ZONES), dtype=bool)
        if boards_path:
            from .hiro_udp import load_boards
            offset = {}
            n = 0
            for s in layout.skins:
                offset[s.name] = n
                n += len(s.sensors)
            for b in load_boards(boards_path, load_layout(layout_path)):
                if b.skin not in offset:
                    continue
                for k, zones in b.masked_zones.items():
                    self.zone_masked[offset[b.skin] + b.layout_sensor(k), zones] = True

    # -- per-episode dressing --

    def set_obstacle(self, i, kind_size, pos, quat=(1, 0, 0, 0)):
        m, d = self.model, self.data
        g = self.obs_geom[i]
        size = np.zeros(3)
        size[:len(kind_size)] = kind_size
        m.geom_size[g] = size
        kind = self.obs_kind[i]
        half = {'box': size, 'sphere': [size[0]] * 3, 'cylinder': [size[0], size[0], size[1]],
                'capsule': [size[0], size[0], size[0] + size[1]]}[kind]
        m.geom_rbound[g] = float(np.linalg.norm(half)) if kind in ('box', 'cylinder') else float(max(half))
        m.geom_aabb[g] = np.concatenate([np.zeros(3), half])
        d.mocap_pos[self.obs_mocap[i]] = pos
        d.mocap_quat[self.obs_mocap[i]] = quat

    def park_all(self):
        for i in range(len(self.obs_geom)):
            self.data.mocap_pos[self.obs_mocap[i]] = PARK
        for mid, _ in self.extra.values():
            self.data.mocap_pos[mid] = PARK

    def set_extra(self, name, size, pos):
        mid, g = self.extra[name]
        self.model.geom_size[g] = size
        self.model.geom_rbound[g] = float(np.linalg.norm(size))
        self.model.geom_aabb[g] = np.concatenate([np.zeros(3), size])
        self.data.mocap_pos[mid] = pos

    def jitter_mounts(self, rng, pos_m=0.002, ang_rad=math.radians(2.0)):
        """Perturb every sensor's mounting; returns (S, 6) [dx dy dz rx ry rz]."""
        offs = []
        for k, (bid, sensors) in enumerate(self.base_sensors):
            new = []
            for xyz, R in sensors:
                dp = rng.uniform(-pos_m, pos_m, 3)
                rv = rng.uniform(-ang_rad, ang_rad, 3)
                q = np.zeros(4)
                mujoco.mju_axisAngle2Quat(q, rv / (np.linalg.norm(rv) + 1e-12), np.linalg.norm(rv))
                dR = np.zeros(9)
                mujoco.mju_quat2Mat(dR, q)
                new.append((xyz + dp, dR.reshape(3, 3) @ R))
                offs.append(np.concatenate([dp, rv]))
            self.caster._sensors[k] = (bid, new)
        return np.array(offs, dtype=np.float32)

    def pose(self, q):
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)

    def clearance(self, obstacles):
        """(K,) min distance from each skin's surface to the active obstacles (m)."""
        out = np.full(len(self.skin_geoms), CLEARANCE_CAP, dtype=np.float32)
        fromto = np.zeros(6)
        for k, geoms in enumerate(self.skin_geoms):
            for o in obstacles:
                og = self.obs_geom[o]
                for g in geoms:
                    dist = mujoco.mj_geomDistance(self.model, self.data, int(g), int(og), CLEARANCE_CAP, fromto)
                    if dist < out[k]:
                        out[k] = dist
        return out


# -- one episode ------------------------------------------------------------------

def _min_jerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def sample_trajectory(world, rng, duration, dt):
    """(T, J) joint positions: min-jerk segments through random waypoints."""
    t = np.arange(0.0, duration, dt)
    n_seg = int(rng.integers(2, 5))
    times = np.concatenate([[0.0], np.sort(rng.uniform(0.15, 0.95, n_seg - 1)) * duration, [duration]])
    wps = rng.uniform(world.q_lo, world.q_hi, (n_seg + 1, len(world.q_lo)))
    q = np.empty((len(t), len(world.q_lo)))
    for i in range(n_seg):
        sel = (t >= times[i]) & (t <= times[i + 1])
        u = _min_jerk((t[sel] - times[i]) / max(times[i + 1] - times[i], 1e-6))
        q[sel] = wps[i] + (wps[i + 1] - wps[i]) * u[:, None]
    return t, q


def _random_shape(rng, kind):
    return {'box': rng.uniform(0.03, 0.20, 3), 'sphere': rng.uniform(0.03, 0.15, 1),
            'cylinder': np.array([rng.uniform(0.02, 0.10), rng.uniform(0.05, 0.40)]),
            'capsule': np.array([rng.uniform(0.03, 0.07), rng.uniform(0.05, 0.20)])}[kind]


def _random_quat(rng):
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def _sweep(world, q_traj, stride):
    """Sensor world poses along the trajectory (every `stride` frames): [(t, [(pos, R)])]."""
    out = []
    for i in range(0, len(q_traj), stride):
        world.pose(q_traj[i])
        out.append((i, [p for skin in world.caster.sensor_world_poses(world.data) for p in skin]))
    return out


def _clear_along(world, q_traj, obstacles, stride, min_clear):
    for i in range(0, len(q_traj), stride):
        world.pose(q_traj[i])
        if world.clearance(obstacles).min() < min_clear:
            return False
    return True


def place_static(world, rng, q_traj, n, min_clear, used, tries=12):
    """Obstacles tight around the limb's path: each in front of a random sensor at a
    random time, at a random clearance, kept only if it never comes within min_clear."""
    stride = max(1, len(q_traj) // 12)
    sweep = _sweep(world, q_traj, stride)
    placed = []
    free = [i for i in range(len(world.obs_geom)) if i not in used]
    for _ in range(n):
        if not free:
            break
        for _attempt in range(tries):
            i = free[int(rng.integers(len(free)))]
            kind = world.obs_kind[i]
            shape = _random_shape(rng, kind)
            _, poses = sweep[int(rng.integers(len(sweep)))]
            origin, R = poses[int(rng.integers(len(poses)))]
            # Somewhere in that sensor's field of view, a random gap beyond the skin.
            cone = math.radians(vl53l5cx.FOV_DEG / 2.0)
            dir_ = R @ np.array([math.tan(rng.uniform(-cone, cone)), math.tan(rng.uniform(-cone, cone)), 1.0])
            dir_ /= np.linalg.norm(dir_)
            reach = float(np.max(shape)) if kind != 'capsule' else float(shape.sum())
            pos = origin + dir_ * (rng.uniform(min_clear + 0.01, 0.45) + reach)
            world.set_obstacle(i, shape, pos, _random_quat(rng))
            if _clear_along(world, q_traj, [i], stride, min_clear):
                placed.append(i)
                free.remove(i)
                break
            world.data.mocap_pos[world.obs_mocap[i]] = PARK
    return placed


def place_world(world, rng, q_traj, min_clear):
    """An optional table under / in front of the limbs and a wall beside them, if clear."""
    stride = max(1, len(q_traj) // 12)
    pts = np.array([p for _, poses in _sweep(world, q_traj, stride) for p, _ in poses])
    lo, hi = pts.min(0), pts.max(0)
    added = []
    if rng.random() < 0.5:
        top = lo[2] - rng.uniform(0.05, 0.35)
        if top > 0.1:
            size = np.array([rng.uniform(0.2, 0.4), rng.uniform(0.25, 0.5), 0.02])
            world.set_extra('table', size, [rng.uniform(0.35, 0.7), rng.uniform(lo[1], hi[1]), top - 0.02])
            added.append('table')
    if rng.random() < 0.4:
        side = 1.0 if (lo[1] + hi[1]) / 2 > 0 else -1.0
        y = (hi[1] if side > 0 else lo[1]) + side * rng.uniform(0.1, 0.6)
        world.set_extra('wall', np.array([0.6, 0.02, 0.8]), [rng.uniform(0.0, 0.4), y, 1.0])
        added.append('wall')
    return added


def plan_moving(world, rng, q_traj, t, used):
    """One obstacle that arrives near (or into) a random sensor's spot at a random time."""
    free = [i for i in range(len(world.obs_geom)) if i not in used]
    i = free[int(rng.integers(len(free)))]
    kind = world.obs_kind[i]
    shape = _random_shape(rng, kind)
    k_arr = int(rng.integers(len(t) // 3, len(t)))
    world.pose(q_traj[k_arr])
    poses = [p for skin in world.caster.sensor_world_poses(world.data) for p in skin]
    origin, R = poses[int(rng.integers(len(poses)))]
    gap = rng.uniform(-0.03, 0.15)           # < 0: the approach ends in contact
    reach = float(np.max(shape)) if kind != 'capsule' else float(shape.sum())
    goal = origin + R[:, 2] * (gap + reach)
    speed = rng.uniform(0.2, 1.0)            # m/s
    approach = _random_quat(rng)[:3]
    approach = approach / np.linalg.norm(approach)
    if approach @ R[:, 2] < 0:
        approach = -approach                 # come from the side the sensor faces
    start = goal + approach * speed * t[k_arr]
    quat = _random_quat(rng)
    hold = rng.random() < 0.5                # stop at the goal, or keep going through
    return i, kind, shape, quat, start, goal, speed, approach, k_arr, hold


def run_episode(world, seed, cfg):
    rng = np.random.default_rng(seed)
    kind = rng.choice(['avoid', 'dynamic', 'free'], p=cfg['mix'])
    dt = 1.0 / world.layout.rate_hz
    t, q_traj = sample_trajectory(world, rng, cfg['duration'], dt)
    world.park_all()
    world.caster._state = {k: np.zeros_like(v) for k, v in world.caster._state.items()}
    world.caster.rng = np.random.default_rng(seed + 1_000_003)
    mount = world.jitter_mounts(rng) if cfg['jitter'] else np.zeros((world.n_sensors, 6), np.float32)
    alive = rng.random(world.n_sensors) >= cfg['dead_p']

    extras = place_world(world, rng, q_traj, cfg['min_clearance']) if kind != 'free' else []
    static = []
    if kind == 'avoid':
        static = place_static(world, rng, q_traj, int(rng.integers(1, cfg['max_static'] + 1)),
                              cfg['min_clearance'], set())
    elif kind == 'dynamic':
        static = place_static(world, rng, q_traj, int(rng.integers(0, 3)), cfg['min_clearance'], set())
    mover = plan_moving(world, rng, q_traj, t, set(static)) if kind == 'dynamic' else None
    if mover:
        world.set_obstacle(mover[0], mover[2], mover[4], mover[3])
    active = static + ([mover[0]] if mover else [])

    T, S, Z, K, O = len(t), world.n_sensors, vl53l5cx.NUM_ZONES, len(world.layout.skins), len(world.obs_geom)
    rec = {
        'readings_mm': np.zeros((T, S, Z), np.uint16), 'raw_mm': np.zeros((T, S, Z), np.uint16),
        'ideal_mm': np.zeros((T, S, Z), np.uint16), 'hit_class': np.zeros((T, S, Z), np.uint8),
        'hit_obstacle': np.full((T, S, Z), -1, np.int8), 'nearest_env_mm': np.zeros((T, S), np.uint16),
        'clearance_m': np.zeros((T, K), np.float32), 'sensor_pos': np.zeros((T, S, 3), np.float32),
        'sensor_quat': np.zeros((T, S, 4), np.float32),
        'obstacles_pos': np.zeros((T, O, 3), np.float32), 'obstacles_vel': np.zeros((T, O, 3), np.float32),
    }
    dists = np.full((Z, 1), -1.0)
    gids = np.full((Z, 1), -1, dtype=np.int32)
    prev_mover = None
    for f in range(T):
        if mover:
            i, _, _, _, start, goal, speed, appr, k_arr, hold = mover
            tt = t[f]
            p = start - appr * speed * tt
            if hold and f >= k_arr:
                p = goal
            world.data.mocap_pos[world.obs_mocap[i]] = p
            vel = np.zeros(3) if prev_mover is None else (p - prev_mover) / dt
            rec['obstacles_vel'][f, i] = vel
            prev_mover = p.copy()
        world.pose(q_traj[f])
        readings, geoms = world.caster.cast(world.data, with_geoms=True)
        s = 0
        for k, (skin, poses) in enumerate(zip(world.layout.skins, world.caster.sensor_world_poses(world.data))):
            r, g, raw = readings[skin.name], geoms[skin.name], world.caster.raw[skin.name]
            for j, (origin, R) in enumerate(poses):
                # The ideal cast again for this sensor, for the labels (cheap: 64 rays).
                d_ray = np.ascontiguousarray(vl53l5cx.ZONE_RAYS @ R.T).reshape(-1, 1)
                dists.fill(-1.0)
                gids.fill(-1)
                _multi_ray(world.model, world.data, np.ascontiguousarray(origin).reshape(3, 1), d_ray, gids,
                           dists, world.caster._sensors[k][0], vl53l5cx.MAX_RANGE_M)
                ideal = vl53l5cx.distances_to_mm(dists.ravel())
                gid = np.where(ideal > 0, gids.ravel(), -1)
                cls = np.where(gid >= 0, world.geom_class[np.maximum(gid, 0)], HIT_NONE)
                obs = np.where(gid >= 0, world.geom_obstacle[np.maximum(gid, 0)], -1)
                cls = np.where(obs >= 0, HIT_STATIC, cls)
                if mover:
                    cls = np.where(obs == mover[0], HIT_MOVING, cls)
                rec['ideal_mm'][f, s], rec['hit_class'][f, s], rec['hit_obstacle'][f, s] = ideal, cls, obs
                env = (ideal > 0) & (cls != HIT_SELF)
                rec['nearest_env_mm'][f, s] = ideal[env].min() if env.any() else 0
                if alive[s]:
                    keep = ~world.zone_masked[s]
                    rec['readings_mm'][f, s] = np.where(keep, r[j], 0)
                    rec['raw_mm'][f, s] = np.where(keep, raw[j], 0)
                rec['sensor_pos'][f, s] = origin
                qw = np.zeros(4)
                mujoco.mju_mat2Quat(qw, R.reshape(9))
                rec['sensor_quat'][f, s] = qw
                s += 1
        rec['clearance_m'][f] = world.clearance(active) if active else CLEARANCE_CAP
        rec['obstacles_pos'][f] = world.data.mocap_pos[world.obs_mocap]
    qd = np.gradient(q_traj, dt, axis=0)
    sizes = world.model.geom_size[world.obs_geom].astype(np.float32)
    quats = world.data.mocap_quat[world.obs_mocap].astype(np.float32)
    act = np.zeros(O, bool)
    act[active] = True
    meta = {
        'seed': int(seed), 'type': str(kind), 'frames': T, 'duration_s': float(t[-1] + dt),
        'static_obstacles': len(static), 'moving': bool(mover), 'world': extras,
        'dead_sensors': int((~alive).sum()), 'masked_zones': int(world.zone_masked.sum()),
        'min_clearance_m': float(rec['clearance_m'].min()),
        'contact_frames': int((rec['clearance_m'].min(axis=1) <= 0).sum()),
        'zones_env_hit': float((rec['hit_class'] >= HIT_STATIC).mean()),
    }
    arrays = dict(rec, t=t.astype(np.float32), q=q_traj.astype(np.float32), qd=qd.astype(np.float32),
                  q_cmd=q_traj.astype(np.float32), sensor_alive=alive, mount_offset=mount,
                  zone_masked=world.zone_masked,
                  obstacles_type=np.array([TYPE_IDS[k] for k in world.obs_kind], np.uint8),
                  obstacles_size=sizes, obstacles_quat=quats, obstacles_active=act,
                  joint_names=np.array(world.joint_names), sensor_names=np.array(world.sensor_names),
                  skin_names=np.array([s.name for s in world.layout.skins]),
                  episode_type=np.array(kind), seed=np.array(seed), schema=np.array(SCHEMA))
    return arrays, meta


# -- running ---------------------------------------------------------------------

_W = None


def _init(args_dict):
    global _W
    _W = (World(args_dict['mjcf'], args_dict['layout'], args_dict['skins'], args_dict['sensor_model'],
                args_dict['boards']), args_dict)


def _work(seed):
    world, a = _W
    path = os.path.join(a['out'], f'episode_{seed:07d}.npz')
    t0 = time.monotonic()
    arrays, meta = run_episode(world, seed, a['cfg'])
    np.savez_compressed(path, **arrays)
    meta['file'] = os.path.basename(path)
    meta['gen_s'] = round(time.monotonic() - t0, 2)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--skins', nargs='+', required=True, help='skin names from the layout')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--start-seed', type=int, default=0, help='episode i uses seed start+i')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--duration', type=float, default=8.0, help='s per episode')
    ap.add_argument('--mix', type=float, nargs=3, default=[0.5, 0.3, 0.2], metavar=('AVOID', 'DYNAMIC', 'FREE'))
    ap.add_argument('--max-static', type=int, default=4)
    ap.add_argument('--min-clearance', type=float, default=0.02, help='m, avoid-episode obstacles')
    ap.add_argument('--dead-p', type=float, default=0.03, help='per-sensor chance of being dead')
    ap.add_argument('--no-jitter', dest='jitter', action='store_false', help='exact sensor mounting')
    ap.add_argument('--ideal-sensor', action='store_true', help='no noise or flicker in readings_mm')
    ap.add_argument('--sensor-model', default=DEFAULT_MODEL)
    ap.add_argument('--layout', default=DEFAULT_LAYOUT)
    ap.add_argument('--boards', default=DEFAULT_BOARDS, help="board map whose masked_zones apply")
    ap.add_argument('--no-masks', dest='masks', action='store_false', help='ignore the masked zones')
    ap.add_argument('--mjcf', default=DEFAULT_MJCF)
    args = ap.parse_args(argv)

    mix = np.array(args.mix, float)
    os.makedirs(args.out, exist_ok=True)
    a = {'mjcf': os.path.abspath(args.mjcf), 'layout': os.path.abspath(args.layout), 'skins': args.skins,
         'boards': os.path.abspath(args.boards) if args.masks else None,
         'sensor_model': None if args.ideal_sensor else os.path.abspath(args.sensor_model), 'out': args.out,
         'cfg': {'duration': args.duration, 'mix': (mix / mix.sum()).tolist(), 'max_static': args.max_static,
                 'min_clearance': args.min_clearance, 'dead_p': args.dead_p, 'jitter': args.jitter}}
    probe = World(a['mjcf'], a['layout'], a['skins'], a['sensor_model'], a['boards'])
    info = {
        'schema': SCHEMA, 'generator': 'golem_proximity_skin.generate', 'config': a['cfg'], 'mjcf': a['mjcf'],
        'skins': args.skins, 'sensors': probe.sensor_names, 'joints': probe.joint_names,
        'rate_hz': probe.layout.rate_hz, 'hit_classes': ['none', 'self', 'static', 'moving', 'world'],
        'obstacle_types': list(TYPE_IDS), 'layout_yaml': open(a['layout'], encoding='utf-8').read(),
        'masked_zones': {n: np.nonzero(m)[0].tolist() for n, m in zip(probe.sensor_names, probe.zone_masked) if m.any()},
        'boards_yaml': open(a['boards'], encoding='utf-8').read() if a['boards'] else None,
        'sensor_model_yaml': open(a['sensor_model'], encoding='utf-8').read() if a['sensor_model'] else None,
    }
    with open(os.path.join(args.out, 'dataset.json'), 'w', newline='\n') as f:
        json.dump(info, f, indent=1)
    if probe.zone_masked.any():
        print(f'masked zones (board map): {int(probe.zone_masked.sum())} across '
              f'{int(probe.zone_masked.any(axis=1).sum())} sensors', flush=True)
    del probe
    seeds = list(range(args.start_seed, args.start_seed + args.episodes))
    print(f'{args.episodes} episodes of {args.duration:g} s, skins {args.skins}, '
          f'{args.workers} workers -> {args.out}', flush=True)
    t0 = time.monotonic()
    done = 0
    with open(os.path.join(args.out, 'manifest.jsonl'), 'a', newline='\n') as man:
        if args.workers <= 1:
            _init(a)
            results = map(_work, seeds)
        else:
            pool = get_context('spawn').Pool(args.workers, initializer=_init, initargs=(a,))
            results = pool.imap_unordered(_work, seeds)
        for meta in results:
            man.write(json.dumps(meta) + '\n')
            man.flush()
            done += 1
            if done % max(1, args.episodes // 20) == 0 or done == args.episodes:
                el = time.monotonic() - t0
                print(f'  {done}/{args.episodes} episodes, {el:.0f} s, '
                      f'{el / done * (args.episodes - done):.0f} s left', flush=True)
        if args.workers > 1:
            pool.close()
            pool.join()


if __name__ == '__main__':
    main()
