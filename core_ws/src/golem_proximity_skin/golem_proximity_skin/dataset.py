"""Load episodes written by golem_proximity_skin.generate.

    from golem_proximity_skin.dataset import Dataset
    ds = Dataset('data/arm_v1')                  # dataset.json + manifest.jsonl + episodes
    ep = ds.episode(0)                           # dict of arrays (see generate.py's docstring)
    for ep in ds.episodes(types=('avoid', 'dynamic')): ...
    pts, sensor = ds.points(ep, frame=10)        # world-frame point cloud of one frame's readings

    python3 -m golem_proximity_skin.dataset summary data/arm_v1

Episodes are separate .npz files, so a training loader can shard, shuffle and
memory-map them as it likes; this module only reads and checks them.
"""
import argparse
import json
import os

import numpy as np

from . import vl53l5cx

INPUTS = ('readings_mm', 'raw_mm', 'q', 'qd')      # what the hardware also has
LABELS = ('ideal_mm', 'hit_class', 'hit_obstacle', 'nearest_env_mm', 'clearance_m')


class Dataset:
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, 'dataset.json'), encoding='utf-8') as f:
            self.info = json.load(f)
        self.meta = []
        with open(os.path.join(root, 'manifest.jsonl'), encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.meta.append(json.loads(line))
        self.meta.sort(key=lambda m: m['seed'])

    def __len__(self):
        return len(self.meta)

    def episode(self, i):
        """Episode i (manifest order, by seed) as a dict of arrays."""
        m = self.meta[i]
        with np.load(os.path.join(self.root, m['file']), allow_pickle=False) as d:
            ep = {k: d[k] for k in d.files}
        if int(ep['schema']) != self.info['schema']:
            raise ValueError(f"{m['file']}: schema {int(ep['schema'])}, dataset {self.info['schema']}")
        return ep

    def episode_arrays(self, i, keys):
        """Only `keys` of episode i (each array is decompressed only when read)."""
        m = self.meta[i]
        with np.load(os.path.join(self.root, m['file']), allow_pickle=False) as d:
            if int(d['schema']) != self.info['schema']:
                raise ValueError(f"{m['file']}: schema {int(d['schema'])}, dataset {self.info['schema']}")
            return {k: d[k] for k in keys}

    def episodes(self, types=None):
        for i, m in enumerate(self.meta):
            if types is None or m['type'] in types:
                yield self.episode(i)

    @staticmethod
    def points(ep, frame, key='readings_mm'):
        """((n, 3) world points, (n,) sensor index) of one frame's non-zero zones."""
        pts, sid = [], []
        for s in range(ep[key].shape[1]):
            r = ep[key][frame, s]
            z = np.nonzero(r != vl53l5cx.NO_TARGET)[0]
            if not z.size:
                continue
            w, x, y, zq = ep['sensor_quat'][frame, s].astype(float)
            R = np.array([
                [1 - 2 * (y * y + zq * zq), 2 * (x * y - zq * w), 2 * (x * zq + y * w)],
                [2 * (x * y + zq * w), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - x * w)],
                [2 * (x * zq - y * w), 2 * (y * zq + x * w), 1 - 2 * (x * x + y * y)]])
            local = vl53l5cx.ZONE_RAYS[z] * (r[z, None] / 1000.0)
            pts.append(local @ R.T + ep['sensor_pos'][frame, s])
            sid.append(np.full(z.size, s))
        if not pts:
            return np.zeros((0, 3)), np.zeros(0, int)
        return np.vstack(pts), np.concatenate(sid)


def summary(root):
    ds = Dataset(root)
    types = {}
    for m in ds.meta:
        types[m['type']] = types.get(m['type'], 0) + 1
    frames = sum(m['frames'] for m in ds.meta)
    contact = sum(1 for m in ds.meta if m['contact_frames'] > 0)
    clear = np.array([m['min_clearance_m'] for m in ds.meta])
    env = np.array([m['zones_env_hit'] for m in ds.meta])
    size = sum(os.path.getsize(os.path.join(root, m['file'])) for m in ds.meta)
    print(f"{root}: {len(ds)} episodes, {frames} frames "
          f"({frames / ds.info['rate_hz'] / 3600:.2f} h at {ds.info['rate_hz']:.1f} Hz), {size / 1e6:.0f} MB")
    print(f"  skins {ds.info['skins']}, {len(ds.info['sensors'])} sensors, joints {len(ds.info['joints'])}")
    print(f'  types {types}; episodes with contact {contact}')
    print(f'  min clearance per episode: median {np.median(clear):.3f} m, '
          f'5th pct {np.percentile(clear, 5):.3f} m')
    print(f'  share of zones hitting a non-robot surface: median {np.median(env):.2f}')
    return ds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=('summary',))
    ap.add_argument('root')
    args = ap.parse_args(argv)
    summary(args.root)


if __name__ == '__main__':
    main()
