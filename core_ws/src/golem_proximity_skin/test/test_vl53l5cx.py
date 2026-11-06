import numpy as np
import pytest

from golem_proximity_skin import vl53l5cx


def test_zone_rays_match_proximity_point_motion():
    # proximity_common.py's ray pre-computation, verbatim.
    half_fov = np.radians(45.0 / 2.0)
    zone_angles = np.linspace(half_fov, -half_fov, 8)
    az, el = np.meshgrid(zone_angles, zone_angles)
    ref = np.stack([(np.sin(az) * np.cos(el)).flatten(), np.sin(el).flatten(),
                    (np.cos(az) * np.cos(el)).flatten()], axis=1)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    assert vl53l5cx.ZONE_RAYS.shape == (64, 3)
    assert np.allclose(vl53l5cx.ZONE_RAYS, ref)
    assert (vl53l5cx.ZONE_RAYS[:, 2] > 0.8).all()       # all within the FOV around +z


def test_distances_to_mm_masks_out_of_range():
    d = np.array([-1.0, 0.01, 0.02, 1.2345, 4.0, 4.5])
    assert vl53l5cx.distances_to_mm(d).tolist() == [0, 0, 20, 1234, 4000, 0]


def test_readings_round_trip():
    r = np.zeros(64, dtype=np.uint16)
    r[10], r[63] = 500, 1500
    pts, zones = vl53l5cx.readings_to_points(r)
    assert zones.tolist() == [10, 63]
    assert np.allclose(np.linalg.norm(pts, axis=1), [0.5, 1.5])
    assert np.allclose(pts[0] / 0.5, vl53l5cx.ZONE_RAYS[10])
    with pytest.raises(ValueError):
        vl53l5cx.readings_to_points(np.zeros(16))
