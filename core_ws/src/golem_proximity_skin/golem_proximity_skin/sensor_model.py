"""VL53L5CX behaviour measured from real recordings, and applied to simulated readings.

    python3 -m golem_proximity_skin.sensor_model fit rec1.npz rec2.npz ... \
        --out config/vl53l5cx_model.yaml

Fits, from any recordings of HIRO's firmware packets (HIRO's own .npz or
hiro_udp's), with no ground truth needed:
  rate_hz              median frame rate per sensor
  noise_sd_mm(d)       temporal noise vs range: the spread of a zone's frame-to-frame
                       change (motion excluded), per raw-range bin
  p_invalid(d)         chance the firmware flags a zone invalid (target_status 255)
                       vs its raw range, per bin
  stay_invalid         P(invalid at t | invalid at t-1): flicker comes in runs
Scenes in the recordings move and are unknown, so this fits noise and dropout,
not bias: a flat target at measured distances is needed for that.

SensorModel.apply() turns ideal ray-cast distances into what the firmware would
send: noisy readings, zones flickering invalid in runs (the firmware still
sends their raw distance), and no return past the sensor's range.
"""
import argparse
import os
from dataclasses import dataclass

import numpy as np
import yaml

from . import vl53l5cx

ZONES = vl53l5cx.NUM_ZONES
FIRMWARE_NO_TARGET_MM = 4000
BINS_MM = [0, 100, 200, 400, 600, 800, 1000, 1300, 1600, 2000, 2500, 3000, 4000]
MOTION_MM = 150          # frame-to-frame change above this is motion, not noise

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, '..', 'config', 'vl53l5cx_model.yaml')


# -- reading recordings -----------------------------------------------------------

def load_frames(path):
    """{sensor id: (t s (n,), raw mm (n, 64), valid (n, 64) bool)} from a recording of
    firmware packets ([id][filtered][raw]); sensors keyed by (file, sensor_ids)."""
    d = np.load(path, allow_pickle=False)
    pk = d['packets']
    if pk.shape[1] < 1 + 4 * ZONES:
        raise ValueError(f'{path}: packets carry no raw distances')
    filt = pk[:, 1:1 + 2 * ZONES].copy().view('<u2')
    raw = pk[:, 1 + 2 * ZONES:1 + 4 * ZONES].copy().view('<u2')
    t = d['timestamps'].astype(np.int64) / 1e9
    out = {}
    for s in np.unique(d['sensor_ids']):
        m = d['sensor_ids'] == s
        order = np.argsort(t[m])
        out[(os.path.basename(path), int(s))] = (
            t[m][order], raw[m][order].astype(np.int32), filt[m][order] < FIRMWARE_NO_TARGET_MM)
    return out


def _bin(d_mm):
    return np.clip(np.digitize(d_mm, BINS_MM) - 1, 0, len(BINS_MM) - 2)


def fit(paths):
    """The model parameters, plus per-bin sample counts, from recordings."""
    nb = len(BINS_MM) - 1
    diffs = [[] for _ in range(nb)]
    valid_n, total_n = np.zeros(nb), np.zeros(nb)
    stay, flips_from_invalid = 0, 0
    periods, frames, sensors = [], 0, 0
    for p in paths:
        for (_, _), (t, raw, valid) in load_frames(p).items():
            sensors += 1
            frames += len(t)
            if len(t) > 2:
                periods.append(np.median(np.diff(t)))
            has = (raw > 0) & (raw < FIRMWARE_NO_TARGET_MM)
            b = _bin(raw)
            for k in range(nb):
                sel = has & (b == k)
                total_n[k] += sel.sum()
                valid_n[k] += (sel & valid).sum()
            both = valid[1:] & valid[:-1] & has[1:] & has[:-1]
            dd = (raw[1:] - raw[:-1])[both]
            mid = ((raw[1:] + raw[:-1]) / 2)[both]
            still = np.abs(dd) < MOTION_MM
            for k, v in zip(_bin(mid[still]), dd[still]):
                diffs[k].append(v)
            inv_prev = ~valid[:-1] & has[:-1]
            stay += (inv_prev & ~valid[1:]).sum()
            flips_from_invalid += inv_prev.sum()
    sd, n_sd = [], []
    for k in range(nb):
        v = np.asarray(diffs[k], dtype=float)
        n_sd.append(int(v.size))
        # Robust sd of one frame: MAD of a difference of two frames / sqrt(2).
        sd.append(float(1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(2)) if v.size >= 50 else None)
    p_inv = [float(1 - valid_n[k] / total_n[k]) if total_n[k] >= 50 else None for k in range(nb)]
    return {
        'source': {'recordings': [os.path.basename(p) for p in paths], 'sensors': sensors, 'frames': frames},
        'rate_hz': float(1.0 / np.median(periods)) if periods else vl53l5cx.RATE_HZ,
        'bins_mm': BINS_MM,
        'noise_sd_mm': sd, 'noise_samples': n_sd,
        'p_invalid': p_inv, 'validity_samples': [int(x) for x in total_n],
        'stay_invalid': float(stay / flips_from_invalid) if flips_from_invalid else 0.0,
    }


# -- applying ---------------------------------------------------------------------

def _fill(values):
    """Bins without enough samples take their nearest filled neighbour's value."""
    v = list(values)
    idx = [i for i, x in enumerate(v) if x is not None]
    if not idx:
        raise ValueError('model has no filled bins')
    return np.array([v[min(idx, key=lambda j: abs(j - i))] for i in range(len(v))], dtype=float)


@dataclass
class SensorModel:
    rate_hz: float
    bins_mm: np.ndarray
    noise_sd_mm: np.ndarray
    p_invalid: np.ndarray
    stay_invalid: float

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL) -> 'SensorModel':
        m = yaml.safe_load(open(path, encoding='utf-8'))
        return cls(float(m['rate_hz']), np.asarray(m['bins_mm'], dtype=float),
                   _fill(m['noise_sd_mm']), _fill(m['p_invalid']), float(m['stay_invalid']))

    def _interp(self, table, d_mm):
        centres = (self.bins_mm[:-1] + self.bins_mm[1:]) / 2
        return np.interp(d_mm, centres, table)

    def new_state(self, shape):
        """Per-zone invalid flags carried between frames (flicker runs)."""
        return np.zeros(shape, dtype=bool)

    def apply(self, dist_m, state, rng):
        """Ideal distances (m, < 0 = no hit; any shape ending in 64) -> (readings mm,
        raw mm), both uint16 as the firmware sends them in this package's convention
        (readings NO_TARGET where flagged invalid or out of range). `state` (from
        new_state, shaped like dist_m) is updated in place.

        A zone flagged invalid stays so with stay_invalid; a valid one turns invalid
        with the rate that keeps its long-run invalid share at p_invalid(d).
        """
        d = np.asarray(dist_m, dtype=float) * 1000.0
        hit = (d >= vl53l5cx.MIN_RANGE_M * 1000) & (d <= vl53l5cx.MAX_RANGE_M * 1000)
        dd = np.where(hit, d, 0.0)
        noisy = dd + rng.standard_normal(d.shape) * self._interp(self.noise_sd_mm, dd)
        p = np.clip(self._interp(self.p_invalid, dd), 0.0, 0.999)
        s = self.stay_invalid
        # Two-state chain with P(stay invalid) = s and stationary share p:
        # P(valid -> invalid) = p (1 - s) / (1 - p).
        p_enter = np.clip(p * (1 - s) / np.maximum(1 - p, 1e-6), 0.0, 1.0)
        u = rng.random(d.shape)
        state[...] = np.where(state, u < s, u < p_enter) & hit
        raw = np.where(hit, np.clip(np.round(noisy), 1, FIRMWARE_NO_TARGET_MM - 1), 0).astype(np.uint16)
        readings = np.where(hit & ~state, raw, vl53l5cx.NO_TARGET).astype(np.uint16)
        return readings, raw


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fit', help='fit the model to recordings')
    f.add_argument('recordings', nargs='+')
    f.add_argument('--out', default=DEFAULT_MODEL)
    args = ap.parse_args(argv)
    m = fit(args.recordings)
    head = ('# VL53L5CX behaviour fitted by `python -m golem_proximity_skin.sensor_model fit`;\n'
            '# see golem_proximity_skin/sensor_model.py. Regenerate, do not edit. Per range bin\n'
            '# (bins_mm edges, raw range): noise_sd_mm (one frame), p_invalid (firmware\n'
            '# target_status 255); null = too few samples. No bias term: needs ground truth.\n')
    with open(args.out, 'w', newline='\n') as fh:
        fh.write(head + yaml.safe_dump(m, sort_keys=False, default_flow_style=None))
    print(f"fitted {m['source']['frames']} frames from {m['source']['sensors']} sensors -> {args.out}")
    print(f"rate {m['rate_hz']:.2f} Hz, stay_invalid {m['stay_invalid']:.3f}")
    for k in range(len(BINS_MM) - 1):
        sd = m['noise_sd_mm'][k]
        pi = m['p_invalid'][k]
        print(f"  {BINS_MM[k]:5d}-{BINS_MM[k + 1]:5d} mm: noise sd "
              f"{'%6.1f mm' % sd if sd is not None else '     -   '} ({m['noise_samples'][k]:>8} diffs) | "
              f"p_invalid {'%.2f' % pi if pi is not None else '  - '} ({m['validity_samples'][k]:>8} zones)")


if __name__ == '__main__':
    main()
