"""The VL53L5CX multizone ToF sensor model the proximity skins carry.

Zone geometry is HIRO-group/proximity_point_motion's (proximity_common.py), so
a cloud built from simulated readings lines up with one built from the
hardware: an 8x8 grid of zones over a 45 deg square field of view, the
boresight on the sensor frame's +z, zone k = row * 8 + col, with azimuth
running from +22.5 deg (col 0) to -22.5 deg (col 7) about the sensor's y axis
and elevation from +22.5 deg (row 0) to -22.5 deg (row 7).

A reading is a distance in mm along the zone's ray; NO_TARGET (0) marks a zone
without a valid return. No ROS imports: the simulator imports this module too.
"""
import numpy as np

RESOLUTION = 8                       # zones per side
NUM_ZONES = RESOLUTION * RESOLUTION
FOV_DEG = 45.0                       # full field of view, both axes
MIN_RANGE_M = 0.02                   # closer returns are reported as NO_TARGET
MAX_RANGE_M = 4.0                    # matches proximity_common.MAX_RANGE
RATE_HZ = 15.0                       # 8x8 mode's maximum ranging frequency
NO_TARGET = 0


def zone_rays(resolution: int = RESOLUTION, fov_deg: float = FOV_DEG) -> np.ndarray:
    """(resolution**2, 3) unit ray directions in the sensor frame, zone-major."""
    half = np.radians(fov_deg / 2.0)
    angles = np.linspace(half, -half, resolution)
    az, el = np.meshgrid(angles, angles)
    rays = np.stack([
        np.sin(az) * np.cos(el),
        np.sin(el),
        np.cos(az) * np.cos(el),
    ], axis=-1).reshape(-1, 3)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


ZONE_RAYS = zone_rays()


def distances_to_mm(dist_m: np.ndarray,
                    min_range: float = MIN_RANGE_M,
                    max_range: float = MAX_RANGE_M) -> np.ndarray:
    """Ray-cast distances (m; negative = miss) to uint16 readings (mm)."""
    d = np.asarray(dist_m, dtype=np.float64)
    valid = (d >= min_range) & (d <= max_range)
    out = np.zeros(d.shape, dtype=np.uint16)
    out[valid] = np.round(d[valid] * 1000.0).astype(np.uint16)
    return out


def readings_to_points(readings_mm: np.ndarray,
                       max_range: float = MAX_RANGE_M) -> tuple:
    """One sensor's NUM_ZONES readings -> (points (k, 3) in the sensor frame, zone ids (k,)).

    Zones reading NO_TARGET or beyond max_range are dropped.
    """
    r = np.asarray(readings_mm, dtype=np.float64).reshape(-1)
    if r.shape[0] != NUM_ZONES:
        raise ValueError(f'expected {NUM_ZONES} zones, got {r.shape[0]}')
    d = r / 1000.0
    keep = (r != NO_TARGET) & (d <= max_range)
    return ZONE_RAYS[keep] * d[keep, None], np.nonzero(keep)[0]
