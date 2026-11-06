import os

import numpy as np
import pytest

from golem_proximity_skin import vl53l5cx
from golem_proximity_skin.layout import (
    Pose, load_layout, matrix_to_quat_xyzw, matrix_to_rpy, parse_layout, rpy_to_matrix)

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'h12_skin.yaml')


def _rotation(R):
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


@pytest.mark.parametrize('rpy', [(0, 0, 0), (0.3, -0.7, 2.1), (-1.463, -3.142, -0.106), (0.2, 1.2, -0.4)])
def test_rpy_round_trip(rpy):
    R = rpy_to_matrix(rpy)
    _rotation(R)
    assert np.allclose(rpy_to_matrix(matrix_to_rpy(R)), R, atol=1e-9)


def test_quat_matches_matrix():
    R = rpy_to_matrix((0.4, -0.2, 1.3))
    x, y, z, w = matrix_to_quat_xyzw(R)
    Rq = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    assert np.allclose(Rq, R, atol=1e-9)


def test_sleeve_sensors_face_out_from_the_axis():
    layout = parse_layout({'skins': [{
        'name': 's', 'parent': 'link',
        'sleeve': {'center': [0.1, 0, -0.2], 'axis': [0, 0, -1], 'ref': [1, 0, 0],
                   'radius': 0.05, 'rings': [-0.02, 0.02], 'per_ring': 4, 'ring_phase_deg': [0, 45]}}]})
    skin = layout.skins[0]
    assert skin.frame == 's_skin' and skin.sensor_frame(3) == 's_sensor_3'
    assert len(skin.sensors) == 8
    _rotation(skin.origin.R)
    assert np.allclose(skin.origin.R[:, 2], [0, 0, -1])
    for s in skin.sensors_in_parent():
        _rotation(s.R)
        radial = s.xyz - np.array([0.1, 0, s.xyz[2]])
        assert np.isclose(np.linalg.norm(radial), 0.05)
        assert np.allclose(s.R[:, 2], radial / 0.05)        # boresight is radially outward


def test_explicit_skin_matches_xacro_form():
    # link4_skin sensor 0 from proximity_point_motion's skins.xacro
    layout = parse_layout({'skins': [{
        'name': 'link4', 'parent': 'fr3_link4',
        'origin': {'xyz': [0, 0, 0], 'rpy': [-1.5708, 0, 0]},
        'sensors': [{'xyz': [-0.141, 0.008, 0.079], 'rpy': [1.684, 0.0, -1.725]}]}]})
    s = layout.skins[0].sensors[0]
    assert np.allclose(s.xyz, [-0.141, 0.008, 0.079])
    # The xacro's normal for that sensor is its frame's +z in the skin frame.
    assert np.allclose(s.R[:, 2], [-0.982, 0.152, -0.113], atol=2e-3)


def test_compose():
    a = Pose.from_xyz_rpy([1, 0, 0], [0, 0, np.pi / 2])
    b = Pose.from_xyz_rpy([1, 0, 0], [0, 0, 0])
    assert np.allclose(a.compose(b).xyz, [1, 1, 0])


def test_rejects_bad_layouts():
    with pytest.raises(ValueError):
        parse_layout({'skins': [{'name': 'a', 'parent': 'l', 'sensors': []}]})
    sensor = [{'xyz': [0, 0, 0], 'rpy': [0, 0, 0]}]
    with pytest.raises(ValueError):
        parse_layout({'skins': [{'name': 'a', 'parent': 'l', 'sensors': sensor},
                                {'name': 'a', 'parent': 'l', 'sensors': sensor}]})


def test_h12_layout_loads():
    layout = load_layout(CONFIG)
    names = [s.name for s in layout.enabled_skins]
    assert len(names) == len(set(names)) > 0
    assert layout.rate_hz <= vl53l5cx.RATE_HZ
    for skin in layout.enabled_skins:
        assert skin.parent.endswith('_link')


def _stl(path):
    import struct
    b = open(path, 'rb').read()
    n = struct.unpack('<I', b[80:84])[0]
    assert 84 + 50 * n == len(b), f'{path}: not a binary STL'
    rec = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    return np.frombuffer(b, dtype=rec, count=n, offset=84)['v'].astype(np.float64)


def test_h12_sensors_sit_on_their_skin_facing_out():
    # Each HIRO skin mesh is in its skin (dermis) frame, as the sensors are; each
    # sensor sits within a few mm of the mesh and looks away from the skin's middle.
    # (Surface normals are no test: around a sensor pocket most faces are its walls.)
    from golem_proximity_skin.layout import resolve_mesh
    for skin in load_layout(CONFIG).enabled_skins:
        assert skin.mesh, skin.name
        tris = _stl(resolve_mesh(skin.mesh))
        cent = tris.mean(axis=1)
        middle = tris.reshape(-1, 3).mean(axis=0)
        for i, s in enumerate(skin.sensors):
            d = np.linalg.norm(cent - s.xyz, axis=1)
            assert d.min() < 0.006, f'{skin.name} sensor {i}: {d.min() * 1000:.1f} mm from its skin'
            assert s.R[:, 2] @ (s.xyz - middle) > 0, f'{skin.name} sensor {i}: looks inward'


def test_select_parents_keeps_one_arm():
    from golem_proximity_skin.layout import select_parents
    right = select_parents(load_layout(CONFIG), 'right_')
    assert right.skins and all(s.parent.startswith('right_') for s in right.skins)
    with pytest.raises(ValueError):
        select_parents(load_layout(CONFIG), 'tail_')
