"""Proximity-skin layout: which skins sit on which robot links, and where their sensors are.

A skin follows HIRO-group/proximity_point_motion's skins.xacro: a dermis frame
`<name>_skin`, fixed to a robot link by `origin`, carrying VL53L5CX sensors at
fixed poses `<name>_sensor_<i>` in the dermis frame, each looking along its +z.

A skin in the layout YAML is given in one of two forms:

  explicit (what skins.xacro holds, one entry per sensor)
    - name: link3
      parent: left_elbow_link
      origin: {xyz: [x, y, z], rpy: [r, p, y]}       # m, rad; skin frame in parent
      sensors:
        - {xyz: [x, y, z], rpy: [r, p, y]}           # sensor frame in skin frame
      mesh: package://.../dermis.stl                 # optional dermis visual (skin frame)
      port: 5012                                     # optional: HIRO firmware UDP port
      label: R5                                      # optional: HIRO label prefix
      enabled: false                                 # optional: drop the skin

  sleeve (a band of sensors around a limb, expanded into the explicit form)
    - name: left_upper_arm
      parent: left_shoulder_yaw_link
      sleeve:
        center: [x, y, z]       # m, parent frame: middle of the band on the limb axis
        axis: [x, y, z]         # limb direction, parent frame -> skin +z
        ref: [x, y, z]          # azimuth 0, parent frame (made orthogonal to axis) -> skin +x
        radius: 0.055           # m, axis to sensor face
        length: 0.12            # m, band length along the axis (dermis visual)
        rings: [-0.04, 0.04]    # m, sensor ring offsets along the axis
        per_ring: 4             # sensors per ring, evenly spaced in azimuth
        ring_phase_deg: [0, 45] # azimuth offset per ring (deg); default 0

Sensors are numbered in order, ring by ring. Frames use URDF conventions (rpy
is fixed-axis roll-pitch-yaw). No ROS imports: the simulator imports this
module too.
"""
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import yaml


def rpy_to_matrix(rpy) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(R) -> tuple:
    R = np.asarray(R, dtype=np.float64)
    sp = -R[2, 0]
    if abs(sp) > 1.0 - 1e-9:     # gimbal lock: put all of the rotation in yaw
        p = np.copysign(np.pi / 2.0, sp)
        return (0.0, float(p), float(np.arctan2(-R[0, 1], R[1, 1])))
    return (float(np.arctan2(R[2, 1], R[2, 2])), float(np.arcsin(sp)),
            float(np.arctan2(R[1, 0], R[0, 0])))


def matrix_to_quat_xyzw(R) -> tuple:
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = 2.0 * np.sqrt(t + 1.0)
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return (float(x), float(y), float(z), float(w))


@dataclass
class Pose:
    """A rigid transform: maps child-frame coordinates into the parent frame."""

    xyz: np.ndarray
    R: np.ndarray

    @classmethod
    def from_xyz_rpy(cls, xyz, rpy) -> 'Pose':
        return cls(np.asarray(xyz, dtype=np.float64).reshape(3), rpy_to_matrix(rpy))

    @property
    def rpy(self) -> tuple:
        return matrix_to_rpy(self.R)

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.xyz
        return T

    def compose(self, child: 'Pose') -> 'Pose':
        return Pose(self.xyz + self.R @ child.xyz, self.R @ child.R)


@dataclass
class Skin:
    name: str
    parent: str                       # robot link (URDF name, no sim prefix)
    origin: Pose                      # skin frame in parent
    sensors: List[Pose]               # sensor frames in skin frame
    mesh: Optional[str] = None
    # Sleeve dimensions, kept for the dermis visual (None for explicit skins).
    radius: Optional[float] = None
    length: Optional[float] = None
    enabled: bool = True
    # HIRO firmware: the UDP port this skin's board sends to, and its label prefix.
    port: Optional[int] = None
    label: Optional[str] = None

    @property
    def frame(self) -> str:
        return f'{self.name}_skin'

    def sensor_frame(self, i: int) -> str:
        return f'{self.name}_sensor_{i}'

    def sensors_in_parent(self) -> List[Pose]:
        return [self.origin.compose(s) for s in self.sensors]


@dataclass
class SkinLayout:
    skins: List[Skin]
    rate_hz: float = 15.0

    @property
    def enabled_skins(self) -> List[Skin]:
        return [s for s in self.skins if s.enabled]

    def sensor_count(self) -> int:
        return sum(len(s.sensors) for s in self.enabled_skins)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError(f'zero-length vector {v.tolist()}')
    return v / n


def expand_sleeve(spec: dict) -> tuple:
    """A sleeve spec -> (skin origin Pose, sensor Poses, radius, length)."""
    axis = _unit(spec['axis'])
    ref = np.asarray(spec.get('ref', [1.0, 0.0, 0.0]), dtype=np.float64)
    ref = _unit(ref - axis * float(ref @ axis))
    R = np.column_stack([ref, np.cross(axis, ref), axis])
    origin = Pose(np.asarray(spec['center'], dtype=np.float64).reshape(3), R)

    radius = float(spec['radius'])
    rings = [float(h) for h in spec.get('rings', [0.0])]
    per_ring = int(spec.get('per_ring', 4))
    phases = spec.get('ring_phase_deg', [0.0] * len(rings))
    if len(phases) != len(rings):
        raise ValueError('ring_phase_deg needs one entry per ring')

    sensors = []
    for h, phase in zip(rings, phases):
        for k in range(per_ring):
            a = np.radians(float(phase)) + 2.0 * np.pi * k / per_ring
            z = np.array([np.cos(a), np.sin(a), 0.0])     # boresight: radially out
            x = np.array([0.0, 0.0, 1.0])                 # along the limb
            sensors.append(Pose(np.array([radius * np.cos(a), radius * np.sin(a), h]),
                                np.column_stack([x, np.cross(z, x), z])))
    return origin, sensors, radius, float(spec.get('length', 2.0 * max(abs(h) for h in rings) + 0.04))


def parse_layout(doc: dict) -> SkinLayout:
    skins = []
    names = set()
    for entry in doc.get('skins', []):
        name = entry['name']
        if name in names:
            raise ValueError(f'duplicate skin name {name!r}')
        names.add(name)
        radius = length = None
        if 'sleeve' in entry:
            origin, sensors, radius, length = expand_sleeve(entry['sleeve'])
        else:
            o = entry.get('origin', {})
            origin = Pose.from_xyz_rpy(o.get('xyz', [0, 0, 0]), o.get('rpy', [0, 0, 0]))
            sensors = [Pose.from_xyz_rpy(s.get('xyz', [0, 0, 0]), s.get('rpy', [0, 0, 0]))
                       for s in entry.get('sensors', [])]
        if not sensors:
            raise ValueError(f'skin {name!r} has no sensors')
        skins.append(Skin(name=name, parent=entry['parent'], origin=origin, sensors=sensors,
                          mesh=entry.get('mesh'), radius=radius, length=length,
                          enabled=bool(entry.get('enabled', True)),
                          port=int(entry['port']) if 'port' in entry else None,
                          label=entry.get('label')))
    return SkinLayout(skins=skins, rate_hz=float(doc.get('rate_hz', 15.0)))


def resolve_mesh(uri: str) -> str:
    """A skin mesh's file path: package://<pkg>/<rel> resolved against this package's
    source tree first (the sim container mounts the source, not an install),
    then the ament index; any other string is returned as is."""
    prefix = 'package://'
    if not uri.startswith(prefix):
        return uri
    pkg, rel = uri[len(prefix):].split('/', 1)
    if pkg == 'golem_proximity_skin':
        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel)
        if os.path.exists(src):
            return src
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory(pkg), rel)
    except Exception as e:
        raise FileNotFoundError(f'cannot resolve {uri}: {e}') from e


def select_parents(layout: SkinLayout, prefix: str) -> SkinLayout:
    """The layout with only the skins whose parent link starts with `prefix` (e.g. 'right_')."""
    skins = [s for s in layout.skins if s.parent.startswith(prefix)]
    if not skins:
        raise ValueError(f'no skin has a parent starting with {prefix!r}')
    return SkinLayout(skins=skins, rate_hz=layout.rate_hz)


def load_layout(path: str) -> SkinLayout:
    with open(path, 'r', encoding='utf-8') as f:
        return parse_layout(yaml.safe_load(f))
