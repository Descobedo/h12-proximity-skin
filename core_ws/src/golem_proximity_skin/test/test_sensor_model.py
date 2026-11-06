import os

import numpy as np
import yaml

from golem_proximity_skin import vl53l5cx
from golem_proximity_skin.hiro_udp import FIRMWARE_NO_TARGET_MM
from golem_proximity_skin.sensor_model import BINS_MM, DEFAULT_MODEL, SensorModel, fit

NB = len(BINS_MM) - 1


def _model(sd=10.0, p=0.2, stay=0.7):
    return SensorModel(12.5, np.array(BINS_MM, float), np.full(NB, sd), np.full(NB, p), stay)


def test_apply_matches_its_parameters():
    m = _model()
    rng = np.random.default_rng(1)
    d = np.full((4000, 64), 1.5)                       # a static target at 1.5 m
    state = m.new_state((64,))
    reads, raws = zip(*(m.apply(row, state, rng) for row in d))
    reads, raws = np.array(reads, float), np.array(raws, float)
    assert abs(raws.std() - 10.0) < 0.5 and abs(raws.mean() - 1500) < 1.0
    inv = reads == vl53l5cx.NO_TARGET
    assert abs(inv.mean() - 0.2) < 0.02                 # long-run invalid share
    prev, nxt = inv[:-1], inv[1:]
    assert abs((prev & nxt).sum() / prev.sum() - 0.7) < 0.03   # runs
    assert (raws > 0).all()                             # invalid zones keep a raw distance


def test_misses_and_out_of_range_are_no_target():
    m = _model(p=0.0)
    r, raw = m.apply(np.array([-1.0, 0.005, 4.5, 1.0]), m.new_state((4,)), np.random.default_rng(0))
    assert r[:3].tolist() == [0, 0, 0] and raw[:3].tolist() == [0, 0, 0] and r[3] > 0


def _recording(path, m, frames=3000, seed=2):
    rng = np.random.default_rng(seed)
    state = m.new_state((64,))
    pk = np.zeros((frames, 257), np.uint8)
    for i in range(frames):
        r, raw = m.apply(np.full(64, 0.9), state, rng)
        filt = np.where(r == 0, FIRMWARE_NO_TARGET_MM, r).astype('<u2')
        pk[i] = np.frombuffer(bytes([0]) + filt.tobytes() + raw.astype('<u2').tobytes(), np.uint8)
    t = (np.arange(frames) * 80_000_000).astype(np.uint64)
    np.savez(path, timestamps=t, packets=pk, sensor_ids=np.zeros(frames, np.uint8),
             skins=np.array(['x'] * frames), skin_indices=np.zeros(frames, np.uint8))


def test_fit_recovers_a_known_sensor(tmp_path):
    true = _model(sd=8.0, p=0.25, stay=0.6)
    _recording(tmp_path / 'r.npz', true)
    got = fit([str(tmp_path / 'r.npz')])
    k = int(np.digitize(900, BINS_MM) - 1)
    assert abs(got['rate_hz'] - 12.5) < 0.01
    assert abs(got['noise_sd_mm'][k] - 8.0) < 0.6
    assert abs(got['p_invalid'][k] - 0.25) < 0.02
    assert abs(got['stay_invalid'] - 0.6) < 0.03


def test_shipped_model_loads():
    assert os.path.exists(DEFAULT_MODEL)
    m = SensorModel.load(DEFAULT_MODEL)
    assert 10.0 < m.rate_hz <= vl53l5cx.RATE_HZ
    assert (m.noise_sd_mm > 0).all() and ((m.p_invalid >= 0) & (m.p_invalid < 1)).all()
    assert 'source' in yaml.safe_load(open(DEFAULT_MODEL))
