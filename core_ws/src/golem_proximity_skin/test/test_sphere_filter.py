import os

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')
from golem_proximity_skin.generate import DEFAULT_MJCF  # noqa: E402
from golem_proximity_skin.sphere_filter import DEFAULT_SPHERES, SphereSelfFilter, parse_spheres  # noqa: E402

if not (os.path.exists(DEFAULT_MJCF) and os.path.exists(DEFAULT_SPHERES)):
    pytest.skip('CL_Assets not checked out', allow_module_level=True)


def test_parses_golems_sphere_model():
    spheres = parse_spheres(DEFAULT_SPHERES)
    links = {p for p, _, _ in spheres}
    assert len(spheres) == 26
    assert {'torso_link', 'right_shoulder_yaw_link', 'right_elbow_link'} <= links
    assert not any('knee' in l or 'hip' in l for l in links)    # the model has no leg spheres
    assert all(r > 0.002 for _, _, r in spheres)


def test_body_is_self_open_space_is_not_and_spheres_follow_the_arm():
    f = SphereSelfFilter(DEFAULT_MJCF)
    f.set_joints([], [])
    m, d = f.model, f.data
    torso = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')]
    assert f.inside(np.array([torso + [0.02, 0.0, 0.3]]))[0]
    assert not f.inside(np.array([[2.0, 0.0, 1.2]]))[0]
    elbow = lambda: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_elbow_link')].copy()  # noqa: E731
    down = elbow()
    f.set_joints(['right_shoulder_pitch_joint'], [-1.4])           # arm raised forward
    up = elbow()
    assert np.linalg.norm(up - down) > 0.1
    assert f.inside(np.array([up]), margin=0.02)[0] and not f.inside(np.array([down]), margin=0.0)[0]
    assert f.clearance(np.array([up]))[0] < 0 < f.clearance(np.array([[2.0, 0.0, 1.2]]))[0]


def test_leg_spheres_cover_the_shin():
    from golem_proximity_skin.sphere_filter import LEG_SPHERES
    f = SphereSelfFilter(DEFAULT_MJCF, extra=(LEG_SPHERES,))
    assert len(f.radius) > 26 and any('knee' in l for l in f.links)
    f.set_joints([], [])
    m, d = f.model, f.data
    # The shin's mesh surface, in the world: what a sensor can actually see of it.
    # (The fit covers the surface; the hollow core may fall between spheres.)
    g = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_BODY, 'right_knee_link') and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    surface = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    assert f.inside(surface).all()
    golem_only = SphereSelfFilter(DEFAULT_MJCF)
    golem_only.set_joints([], [])
    assert golem_only.inside(surface).mean() < 0.05            # GOLEM's model leaves the shin bare
