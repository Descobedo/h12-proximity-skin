"""Ray-cast VL53L5CX readings for a skin layout on a MuJoCo model.

Used by the RoboCasa sim (golem_robocasa/tof_skin_sensors.py) and the offline
preview (golem_proximity_skin.preview). No ROS imports.

Each zone is one ray from the sensor origin along its zone direction
(vl53l5cx.ZONE_RAYS); its reading is the first hit's distance, then, given a
sensor_model.SensorModel, that distance as the real sensor reports it (noise,
zones flickering invalid). The sensor's
own parent link is excluded, as the lidar excludes the torso it sits in, so a
sleeve sensor sitting on its link's shell does not see that shell. Every other
geom, the robot's other links included, is a target: the real skin sees them
too.
"""
import mujoco
import numpy as np

from . import vl53l5cx
from .layout import SkinLayout

_GEOMGROUP_ALL = np.ones((6, 1), dtype=np.uint8)
_MULTIRAY_HAS_NORMAL = 'normal' in (mujoco.mj_multiRay.__doc__ or '')


def _multi_ray(model, data, origin, dirs_world, geomids, dists, bodyexclude, cutoff):
    # MuJoCo >= 3.4 added a `normal` output between dist and nray.
    args = [model, data, origin, dirs_world, _GEOMGROUP_ALL, 1, bodyexclude, geomids, dists]
    if _MULTIRAY_HAS_NORMAL:
        args.append(None)
    mujoco.mj_multiRay(*args, dirs_world.shape[0] // 3, cutoff)


def resolve_body(model, name: str, prefixes=('',)) -> int:
    """Body id of `name` under the first prefix that names one (e.g. robosuite's 'robot0_')."""
    for p in prefixes:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{p}{name}')
        if bid >= 0:
            return int(bid)
    raise KeyError(f"body {name!r} not found in the model (prefixes {list(prefixes)})")


class ToFSkinCaster:
    """Casts every enabled sensor of a layout against an MjData."""

    def __init__(self, model, layout: SkinLayout, body_prefixes=('',),
                 max_range: float = vl53l5cx.MAX_RANGE_M, sensor_model=None, seed=None):
        self.model = model
        self.sensor_model = sensor_model
        self.rng = np.random.default_rng(seed)
        self.raw = {}             # skin -> (n_sensors, 64) uint16 raw mm of the last cast
        self.max_range = float(max_range)
        self.skins = layout.enabled_skins
        self._sensors = []     # per skin: (parent body id, [(xyz, R) of each sensor in parent])
        for skin in self.skins:
            bid = resolve_body(model, skin.parent, body_prefixes)
            self._sensors.append((bid, [(p.xyz, p.R) for p in skin.sensors_in_parent()]))
        n = vl53l5cx.NUM_ZONES
        self._state = {s.name: np.zeros((len(s.sensors), n), dtype=bool) for s in self.skins}
        self._dists = np.full((n, 1), -1.0)
        self._geomids = np.full((n, 1), -1, dtype=np.int32)

    def sensor_world_poses(self, data):
        """Per skin, per sensor: (origin (3,), R (3, 3)) in the world frame."""
        out = []
        for bid, sensors in self._sensors:
            bp = data.xpos[bid]
            bR = data.xmat[bid].reshape(3, 3)
            out.append([(bp + bR @ xyz, bR @ R) for xyz, R in sensors])
        return out

    def cast(self, data, with_geoms: bool = False):
        """{skin name: (n_sensors, 64) uint16 readings in mm, NO_TARGET = no return}.

        With with_geoms, also {skin name: (n_sensors, 64) int32 id of the geom
        each zone hit, -1 for none}: the sim knows what it hit, the hardware
        does not. Reads only `data`; pass a snapshot to cast while physics runs on.
        """
        readings, geoms = {}, {}
        for skin, (bid, _), poses in zip(self.skins, self._sensors, self.sensor_world_poses(data)):
            out = np.zeros((len(poses), vl53l5cx.NUM_ZONES), dtype=np.uint16)
            raw = np.zeros_like(out)
            gids = np.full((len(poses), vl53l5cx.NUM_ZONES), -1, dtype=np.int32)
            for i, (origin, R) in enumerate(poses):
                dirs = np.ascontiguousarray(vl53l5cx.ZONE_RAYS @ R.T).reshape(-1, 1)
                self._dists.fill(-1.0)
                self._geomids.fill(-1)
                _multi_ray(self.model, data, np.ascontiguousarray(origin).reshape(3, 1),
                           dirs, self._geomids, self._dists, bid, self.max_range)
                if self.sensor_model is None:
                    out[i] = vl53l5cx.distances_to_mm(self._dists.ravel(), max_range=self.max_range)
                    raw[i] = out[i]
                else:
                    out[i], raw[i] = self.sensor_model.apply(
                        self._dists.ravel(), self._state[skin.name][i], self.rng)
                gids[i] = np.where(out[i] != vl53l5cx.NO_TARGET, self._geomids.ravel(), -1)
            readings[skin.name] = out
            self.raw[skin.name] = raw
            geoms[skin.name] = gids
        return (readings, geoms) if with_geoms else readings
