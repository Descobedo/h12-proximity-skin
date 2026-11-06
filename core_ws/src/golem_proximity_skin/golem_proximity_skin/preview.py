"""Preview a skin layout on the H1-2 in MuJoCo, without ROS or the sim containers.

    python3 -m golem_proximity_skin.preview \
        --mjcf CL_Assets/mujoco_assets/scene_h1_2_magpie.xml \
        --layout core_ws/src/golem_proximity_skin/config/h12_skin.yaml \
        --out skin_preview.png            # or --viewer for the interactive viewer

Adds the dermis sleeves and sensor boards to the model as visual-only geoms,
places a few obstacles around the robot, ray-casts every sensor with the sim's
caster (mujoco_tof.ToFSkinCaster) and draws each zone's return as a dot
coloured by its sensor, with each sensor's field of view out to its nearest
return. Prints per-skin statistics.
"""
import argparse
import colorsys
import math

import mujoco
import numpy as np

from . import vl53l5cx
from .layout import load_layout, resolve_mesh
from .mujoco_tof import ToFSkinCaster

SKIN_RGBA = [0.25, 0.7, 0.9, 0.35]
BOARD_RGBA = [0.1, 0.1, 0.12, 1.0]

# A reaching pose (rad) so the arm sleeves look at the obstacles.
POSE = {
    'left_shoulder_pitch_joint': -0.6, 'left_shoulder_roll_joint': 0.25, 'left_elbow_joint': 0.3,
    'right_shoulder_pitch_joint': -0.9, 'right_shoulder_roll_joint': -0.15, 'right_elbow_joint': 0.6,
}

OBSTACLES = [   # (name, type, size, pos, rgba)
    ('table', mujoco.mjtGeom.mjGEOM_BOX, [0.3, 0.5, 0.02], [0.65, 0.0, 0.85], [0.6, 0.45, 0.3, 1]),
    ('post_left', mujoco.mjtGeom.mjGEOM_CYLINDER, [0.04, 0.6, 0], [0.25, 0.45, 0.6], [0.8, 0.8, 0.8, 1]),
    ('ball_right', mujoco.mjtGeom.mjGEOM_SPHERE, [0.08, 0, 0], [0.45, -0.35, 1.1], [0.9, 0.3, 0.3, 1]),
    ('crate', mujoco.mjtGeom.mjGEOM_BOX, [0.15, 0.15, 0.25], [0.35, -0.05, 0.25], [0.3, 0.5, 0.3, 1]),
]


def _quat(R) -> list:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return q.tolist()


def build_spec(mjcf: str, layout):
    """The MJCF with each enabled skin's dermis (its mesh, else its sleeve) and sensor
    boards as visual-only geoms (named <skin>_dermis, <skin>_board_<i>)."""
    spec = mujoco.MjSpec.from_file(mjcf)
    bodies = {b.name: b for b in spec.bodies}
    for skin in layout.enabled_skins:
        body = bodies[skin.parent]
        o = skin.origin
        if skin.mesh:
            spec.add_mesh(name=f'{skin.name}_dermis_mesh', file=resolve_mesh(skin.mesh))
            body.add_geom(name=f'{skin.name}_dermis', type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=f'{skin.name}_dermis_mesh', pos=o.xyz.tolist(), quat=_quat(o.R),
                          rgba=SKIN_RGBA, contype=0, conaffinity=0, group=2)
        elif skin.radius is not None:
            body.add_geom(name=f'{skin.name}_dermis', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                          size=[skin.radius - 0.004, skin.length / 2.0, 0], pos=o.xyz.tolist(),
                          quat=_quat(o.R), rgba=SKIN_RGBA, contype=0, conaffinity=0, group=2)
        for i, s in enumerate(skin.sensors_in_parent()):
            body.add_geom(name=f'{skin.name}_board_{i}', type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[0.008, 0.006, 0.002], pos=s.xyz.tolist(), quat=_quat(s.R),
                          rgba=BOARD_RGBA, contype=0, conaffinity=0, group=2)
    return spec


def build_model(mjcf: str, layout, with_obstacles: bool = True):
    spec = build_spec(mjcf, layout)
    if with_obstacles:
        for name, gtype, size, pos, rgba in OBSTACLES:
            spec.worldbody.add_geom(name=name, type=gtype, size=size, pos=pos, rgba=rgba)
    return spec.compile()


def pose_robot(model, data, pelvis_fixed: bool = True):
    for name, q in POSE.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = q
    mujoco.mj_forward(model, data)


def sensor_hues(n):
    return [colorsys.hsv_to_rgb(k / max(n, 1), 0.85, 1.0) for k in range(n)]


def add_geom(scene, gtype, size, pos, mat, rgba, cap=None):
    """Append one decor geom to an mjvScene; None when the scene is full."""
    if scene.ngeom >= (cap or scene.maxgeom):
        return None
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, dtype=np.float64), np.asarray(pos, dtype=np.float64),
                        np.asarray(mat, dtype=np.float64).reshape(9), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1
    return g


def add_dot(scene, pos, radius, rgba):
    return add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0], pos, np.eye(3), rgba)


def add_line(scene, a, b, rgba, width=2.0):
    g = add_geom(scene, mujoco.mjtGeom.mjGEOM_LINE, [1, 1, 1], np.zeros(3), np.eye(3), rgba)
    if g is not None:
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width,
                             np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return g


def fov_corners() -> np.ndarray:
    """The four corner directions of a VL53L5CX's field of view, scaled to z = 1."""
    half = math.radians(vl53l5cx.FOV_DEG / 2.0)
    return np.array([[math.tan(sx * half), math.tan(sy * half), 1.0]
                     for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def add_fov(scene, origin, R, depth, rgba, width=2.0):
    ends = [origin + R @ (c * depth) for c in fov_corners()]
    for j in range(4):
        add_line(scene, origin, ends[j], rgba, width)
        add_line(scene, ends[j], ends[(j + 1) % 4], rgba, width)


def overlay(scene, caster, data, readings, fov_draw_m=0.35, max_geoms=None):
    """Add zone returns and FOV edges (drawn out to the nearest return, at most
    fov_draw_m) to an mjvScene."""
    half = math.radians(vl53l5cx.FOV_DEG / 2.0)
    corners = np.array([[math.tan(sx * half), math.tan(sy * half), 1.0]
                        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
    poses = caster.sensor_world_poses(data)
    n_total = sum(len(p) for p in poses)
    hues = sensor_hues(n_total)
    k = 0
    cap = max_geoms or scene.maxgeom

    def add(gtype, size, pos, mat, rgba):
        return add_geom(scene, gtype, size, pos, mat, rgba, cap)

    for skin, skin_poses in zip(caster.skins, poses):
        r_mm = readings[skin.name]
        for (origin, R), r in zip(skin_poses, r_mm):
            hue = hues[k]
            k += 1
            pts, _ = vl53l5cx.readings_to_points(r)
            for p in pts:
                add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.006, 0, 0], origin + R @ p, np.eye(3), [*hue, 1.0])
            valid = r[r != vl53l5cx.NO_TARGET]
            depth = min(float(valid.min()) / 1000.0, fov_draw_m) if valid.size else 0.08
            rgba = [*hue, 0.9] if valid.size else [0.6, 0.6, 0.6, 0.4]
            ends = [origin + R @ (c * depth) for c in corners]
            for a, b in [(origin, e) for e in ends] + [(ends[j], ends[(j + 1) % 4]) for j in range(4)]:
                g = add(mujoco.mjtGeom.mjGEOM_LINE, [1, 1, 1], np.zeros(3), np.eye(3), rgba)
                if g is not None:
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2.0,
                                         np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def report(readings):
    for name, r in readings.items():
        hits = r != vl53l5cx.NO_TARGET
        near = (r[hits].min() / 1000.0) if hits.any() else float('nan')
        print(f'  {name:16s} {r.shape[0]} sensors, {int(hits.sum()):4d}/{r.size} zones with a return, '
              f'nearest {near:.3f} m')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mjcf', required=True, help='H1-2 scene MJCF (CL_Assets/mujoco_assets/scene_h1_2_*.xml)')
    ap.add_argument('--layout', required=True, help='skin layout YAML')
    ap.add_argument('--out', default='skin_preview.png', help='rendered image path')
    ap.add_argument('--viewer', action='store_true', help='open the interactive viewer instead')
    ap.add_argument('--no-obstacles', action='store_true')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=960)
    ap.add_argument('--azimuth', type=float, default=150.0)
    ap.add_argument('--elevation', type=float, default=-20.0)
    ap.add_argument('--distance', type=float, default=2.4)
    ap.add_argument('--lookat', type=float, nargs=3, default=[0.15, 0.0, 0.95], help='camera target, m')
    ap.add_argument('--fov-draw', type=float, default=0.35, help='m, longest FOV edge drawn')
    args = ap.parse_args(argv)

    layout = load_layout(args.layout)
    model = build_model(args.mjcf, layout, not args.no_obstacles)
    data = mujoco.MjData(model)
    pose_robot(model, data)
    caster = ToFSkinCaster(model, layout)
    readings = caster.cast(data)
    print(f'{len(caster.skins)} skins, {layout.sensor_count()} sensors')
    report(readings)

    if args.viewer:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as v:
            while v.is_running():
                readings = caster.cast(data)
                with v.lock():
                    v.user_scn.ngeom = 0
                    overlay(v.user_scn, caster, data, readings, args.fov_draw)
                v.sync()
        return

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(model, args.height, args.width, max_geom=20000)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = args.lookat
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance
    renderer.update_scene(data, cam)
    overlay(renderer.scene, caster, data, readings, args.fov_draw)
    img = renderer.render()
    from PIL import Image
    Image.fromarray(img).save(args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
