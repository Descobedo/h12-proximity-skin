import os

import numpy as np
import pytest

pytest.importorskip('mujoco')
from golem_proximity_skin import generate as G  # noqa: E402
from golem_proximity_skin.dataset import Dataset  # noqa: E402

if not os.path.exists(G.DEFAULT_MJCF):
    pytest.skip('CL_Assets not checked out', allow_module_level=True)

CFG = {'duration': 1.6, 'mix': [1.0, 0.0, 0.0], 'max_static': 3, 'min_clearance': 0.02,
       'dead_p': 0.0, 'jitter': True}


@pytest.fixture(scope='module')
def world():
    return G.World(G.DEFAULT_MJCF, G.DEFAULT_LAYOUT, ['right_upper_arm', 'right_forearm'], G.DEFAULT_MODEL)


def test_avoid_episode_is_clear_and_well_formed(world):
    arr, meta = G.run_episode(world, 3, CFG)
    T, S = arr['readings_mm'].shape[:2]
    assert meta['type'] == 'avoid' and T == meta['frames'] and S == len(world.sensor_names)
    assert arr['q'].shape == (T, len(world.joint_names))
    assert (arr['q'] >= world.q_lo - 1e-6).all() and (arr['q'] <= world.q_hi + 1e-6).all()
    assert arr['clearance_m'].min() >= CFG['min_clearance'] - 1e-3
    hit = arr['ideal_mm'] > 0
    assert ((arr['hit_class'] > 0) == hit).all()
    valid = arr['readings_mm'] > 0
    assert (hit | ~valid).all()                       # no reading without a true return
    err = arr['readings_mm'].astype(float)[valid] - arr['ideal_mm'][valid]
    assert abs(np.median(err)) < 5


def test_same_seed_same_episode(world):
    a, _ = G.run_episode(world, 11, dict(CFG, mix=[0.0, 1.0, 0.0]))
    b, _ = G.run_episode(world, 11, dict(CFG, mix=[0.0, 1.0, 0.0]))
    for k in ('readings_mm', 'raw_mm', 'q', 'obstacles_pos', 'clearance_m'):
        assert np.array_equal(a[k], b[k]), k


def test_cli_writes_a_loadable_dataset(tmp_path):
    G.main(['--out', str(tmp_path), '--skins', 'right_knee', '--episodes', '2', '--workers', '1',
            '--duration', '1.2'])
    ds = Dataset(str(tmp_path))
    assert len(ds) == 2 and ds.info['skins'] == ['right_knee']
    ep = ds.episode(0)
    assert ep['joint_names'][3] == 'right_knee_joint'
    pts, sid = Dataset.points(ep, 0, 'ideal_mm')
    assert pts.shape[0] == sid.shape[0] == int((ep['ideal_mm'][0] > 0).sum())


def test_board_masks_apply_to_generated_readings(tmp_path):
    boards = tmp_path / 'boards.yaml'
    boards.write_text('boards:\n  - skin: right_upper_arm\n    sensors: [5, 4, 3, 0]\n'
                      '    masked_zones: {1: [0, 1, 2]}\n')
    w = G.World(G.DEFAULT_MJCF, G.DEFAULT_LAYOUT, ['right_upper_arm'], G.DEFAULT_MODEL, str(boards))
    arr, meta = G.run_episode(w, 4, dict(CFG, mix=[0.0, 0.0, 1.0]))
    s = w.sensor_names.index('right_upper_arm_sensor_4')          # firmware 1 is layout sensor 4
    assert arr['zone_masked'][s, :3].all() and arr['zone_masked'].sum() == 3 and meta['masked_zones'] == 3
    assert (arr['readings_mm'][:, s, :3] == 0).all() and (arr['raw_mm'][:, s, :3] == 0).all()
