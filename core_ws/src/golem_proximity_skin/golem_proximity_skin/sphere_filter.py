"""Geometric self-filter: a sensed point inside the robot's sphere model is the robot itself.

The method of HIRO-group/proximity_point_motion's self_filter.py (branch nn_test):
a sphere-packed robot model (each sphere a `<link>_sphereN` URDF link fixed to its
kinematic link), placed by forward kinematics at the robot's joint angles; a point
within (radius + margin) of any sphere centre is a self point. Here the spheres are
GOLEM's H1-2 self-collision model (CL_Assets/ros_assets/h1_2_magpie_sphere.urdf,
the one golem_upper_ik uses), and the kinematics are MuJoCo's on the same MJCF the
generator uses, so the spheres sit in exactly the frame of a generated episode.

    f = SphereSelfFilter(mjcf, sphere_urdf)
    f.set_joints(joint_names, q)            # the joints not named stay at zero
    self_mask = f.inside(points_world, margin=0.02)
"""
import os
import re

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPHERES = os.path.join(HERE, '..', '..', '..', '..', 'CL_Assets', 'ros_assets', 'h1_2_magpie_sphere.urdf')
LEG_SPHERES = os.path.join(HERE, '..', 'config', 'h12_leg_spheres.yaml')   # scripts/fit_spheres.py

_LINK = re.compile(r'<link name="([^"]+_sphere\d+)"\s*>(.*?)</link>', re.S)
_RADIUS = re.compile(r'<sphere radius="([^"]+)"')
_JOINT = re.compile(r'<joint[^>]*type="fixed"[^>]*>(.*?)</joint>', re.S)
_PARENT = re.compile(r'<parent link="([^"]+)"')
_CHILD = re.compile(r'<child link="([^"]+)"')
_ORIGIN = re.compile(r'<origin([^>]*)/?>')
_ATTR = re.compile(r'(xyz|rpy)="([^"]*)"')


def parse_spheres(path):
    """[(parent link, centre in the link frame (3,), radius)] from a sphere-packed URDF.

    Regex, not an XML parser: GOLEM's file carries Drake-namespaced tags with no
    namespace declaration, which ElementTree refuses.
    """
    text = open(path, encoding='utf-8').read()
    radius = {}
    for name, body in _LINK.findall(text):
        r = _RADIUS.search(body)
        if r:
            radius[name] = float(r.group(1))
    out = []
    for body in _JOINT.findall(text):
        parent, child, origin = _PARENT.search(body), _CHILD.search(body), _ORIGIN.search(body)
        if not (parent and child) or child.group(1) not in radius:
            continue
        attrs = dict(_ATTR.findall(origin.group(1))) if origin else {}
        xyz = np.array([float(v) for v in attrs.get('xyz', '0 0 0').split()])
        if any(float(v) for v in attrs.get('rpy', '0 0 0').split()):
            raise ValueError(f'{path}: {child.group(1)} has a rotated origin (a sphere needs none)')
        if radius[child.group(1)] > 0.002:              # MorphIt's 1 mm dummies are not body
            out.append((parent.group(1), xyz, radius[child.group(1)]))
    if not out:
        raise ValueError(f'{path}: no spheres found')
    return out


def load_extra(path):
    """[(link, centre, radius)] from a spheres YAML (scripts/fit_spheres.py)."""
    import yaml
    doc = yaml.safe_load(open(path, encoding='utf-8'))
    return [(s['link'], np.asarray(s['xyz'], dtype=float), float(s['radius'])) for s in doc['spheres']]


class SphereSelfFilter:
    """GOLEM's sphere model, plus `extra` sphere files (e.g. LEG_SPHERES) if given."""

    def __init__(self, mjcf, sphere_urdf=DEFAULT_SPHERES, extra=()):
        self.model = mujoco.MjModel.from_xml_path(mjcf)
        self.data = mujoco.MjData(self.model)
        spheres = parse_spheres(sphere_urdf)
        for path in extra:
            spheres += load_extra(path)
        self.body = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, p) for p, _, _ in spheres])
        missing = [p for (p, _, _), b in zip(spheres, self.body) if b < 0]
        if missing:
            raise KeyError(f'sphere parents not in the MJCF: {sorted(set(missing))}')
        self.local = np.array([c for _, c, _ in spheres])
        self.radius = np.array([r for _, _, r in spheres])
        self.links = [p for p, _, _ in spheres]
        self.centres = None
        self._qadr = {}

    def set_joints(self, joint_names, q):
        """Place the spheres with the named joints at q (rad); every other joint at zero."""
        d = self.data
        d.qpos[:] = self.model.qpos0
        for name, v in zip(joint_names, q):
            if name not in self._qadr:
                j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
                if j < 0:
                    raise KeyError(f'no joint {name!r}')
                self._qadr[name] = self.model.jnt_qposadr[j]
            d.qpos[self._qadr[name]] = v
        mujoco.mj_kinematics(self.model, d)
        R = d.xmat[self.body].reshape(-1, 3, 3)
        self.centres = d.xpos[self.body] + np.einsum('mij,mj->mi', R, self.local)

    def clearance(self, points):
        """(N,) distance (m) from each world point to the nearest sphere surface; < 0 inside."""
        if self.centres is None:
            raise RuntimeError('set_joints first')
        d = np.sqrt(((points[:, None, :] - self.centres[None, :, :]) ** 2).sum(-1))
        return (d - self.radius[None, :]).min(axis=1)

    def inside(self, points, margin=0.0):
        """(N,) bool: points (N, 3), world frame, within radius + margin of a sphere."""
        if self.centres is None:
            raise RuntimeError('set_joints first')
        d2 = ((points[:, None, :] - self.centres[None, :, :]) ** 2).sum(-1)
        return (d2 <= (self.radius + margin)[None, :] ** 2).any(axis=1)
