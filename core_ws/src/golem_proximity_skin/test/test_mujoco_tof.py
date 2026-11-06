import numpy as np
import pytest

from golem_proximity_skin import vl53l5cx
from golem_proximity_skin.layout import parse_layout

mujoco = pytest.importorskip('mujoco')
from golem_proximity_skin.mujoco_tof import ToFSkinCaster  # noqa: E402

# A capsule "limb" with a skin, a wall 0.5 m in front of it along +x.
MJCF = """
<mujoco>
  <worldbody>
    <body name="robot0_arm_link" pos="0 0 1">
      <geom type="capsule" size="0.04 0.1"/>
    </body>
    <geom name="wall" type="box" size="0.01 1 1" pos="0.55 0 1"/>
  </worldbody>
</mujoco>
"""

LAYOUT = {'skins': [{
    'name': 'arm', 'parent': 'arm_link',
    'sleeve': {'center': [0, 0, 0], 'axis': [0, 0, 1], 'ref': [1, 0, 0],
               'radius': 0.05, 'rings': [0.0], 'per_ring': 4}}]}


def test_cast_sees_the_wall_and_not_its_own_link():
    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    caster = ToFSkinCaster(model, parse_layout(LAYOUT), body_prefixes=('robot0_', ''))
    r = caster.cast(data)['arm']
    assert r.shape == (4, vl53l5cx.NUM_ZONES)
    # Sensor 0 looks along +x from x = 0.05: the wall face is at x = 0.54.
    front = r[0]
    assert (front != vl53l5cx.NO_TARGET).all()
    centre = np.argmax(vl53l5cx.ZONE_RAYS[:, 2])
    assert abs(int(front[centre]) - 490) <= 12
    expected = 490.0 / vl53l5cx.ZONE_RAYS[:, 2]          # flat wall: d = 0.49 / cos
    assert np.allclose(front, expected, atol=2)
    # The others face away (+y, -x, -y) into empty space; their own capsule is excluded.
    assert (r[1:] == vl53l5cx.NO_TARGET).all()


def test_unknown_parent_is_an_error():
    model = mujoco.MjModel.from_xml_string(MJCF)
    bad = {'skins': [dict(LAYOUT['skins'][0], parent='no_such_link')]}
    with pytest.raises(KeyError):
        ToFSkinCaster(model, parse_layout(bad))


def test_cast_reports_the_geom_each_zone_hit():
    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    caster = ToFSkinCaster(model, parse_layout(LAYOUT), body_prefixes=('robot0_', ''))
    readings, geoms = caster.cast(data, with_geoms=True)
    wall = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'wall')
    assert (geoms['arm'][0] == wall).all()
    assert (geoms['arm'][1:] == -1).all()
    assert np.array_equal(readings['arm'], caster.cast(data)['arm'])
