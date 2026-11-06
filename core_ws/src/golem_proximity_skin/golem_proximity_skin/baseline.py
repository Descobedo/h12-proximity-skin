"""Baseline skin perception model: self-return filtering and per-skin clearance.

    python3 -m golem_proximity_skin.baseline train data/arm_v1 --out models/arm_v1
    python3 -m golem_proximity_skin.baseline eval  data/arm_v1 --model models/arm_v1

From only what the hardware gives, one frame at a time (every sensor's 8x8
readings, raw distances and validity, plus the moved joints' angles and
velocities), it predicts:
  zone class   per zone: none (no return), self (the robot's own body), env
               (anything else); what a real skin needs to stop reacting to itself
  clearance    per skin: distance from its link to the nearest obstacle (m, capped
               at 1), from which "is anything within 10 cm" follows

Model: a small CNN shared by all sensors encodes each sensor's 8x8 frame; a
learned embedding says which sensor it is; an MLP encodes the joints. Their
concatenation gives per-zone class logits (1x1 conv) and, pooled over every
sensor, the clearances.

Compared, on held-out episodes, against what needs no learning:
  all-env         every valid return is environment (what simulate.py reacts to)
  self-threshold  valid returns under a distance (best on validation) are self
  sphere filter   HIRO's geometric self-filter (sphere_filter.py): a valid return inside
                  GOLEM's H1-2 sphere model, placed by the joint angles, plus a margin
                  (best on validation), is self
  nearest-reading clearance = the skin's nearest valid reading, all of it, or with
                  the true self-returns removed (an oracle: the best that filtering
                  alone could do)
Episodes are split by seed: the last 10% test, the 10% before them validation.
Needs torch (CPU is fine).
"""
import argparse
import json
import math
import os
import time

import numpy as np

from .dataset import Dataset

NONE, SELF, ENV = 0, 1, 2
CLASS_NAMES = ('none', 'self', 'env')
NEAR_M = 0.10
CAP_M = 1.0
NORM_MM = 4000.0


# -- data ----------------------------------------------------------------------

def zone_class(hit_class):
    """generate.py's hit_class -> NONE / SELF / ENV."""
    return np.where(hit_class == 0, NONE, np.where(hit_class == 1, SELF, ENV)).astype(np.int64)


class Frames:
    """Every frame of some episodes, kept compact (uint16) until a batch is drawn.

    Reads only the arrays training uses, straight into buffers sized from the
    manifest, so loading costs one decompression per array and no copies.
    """

    def __init__(self, ds, indices, log_every=100):
        metas = [ds.meta[i] for i in indices]
        self.n = sum(m['frames'] for m in metas)
        first = ds.episode_arrays(indices[0], ('readings_mm', 'clearance_m', 'q'))
        S, K, J = first['readings_mm'].shape[1], first['clearance_m'].shape[1], first['q'].shape[1]
        self.readings = np.empty((self.n, S, 64), np.uint16)
        self.raw = np.empty((self.n, S, 64), np.uint16)
        self.cls = np.empty((self.n, S, 64), np.uint8)
        self.clear = np.empty((self.n, K), np.float32)
        self.q = np.empty((self.n, J), np.float32)
        self.qd = np.empty((self.n, J), np.float32)
        self.alive = np.empty((self.n, S), np.float32)
        t0, at = time.monotonic(), 0
        for k, i in enumerate(indices):
            ep = ds.episode_arrays(i, ('readings_mm', 'raw_mm', 'hit_class', 'clearance_m', 'q', 'qd',
                                       'sensor_alive'))
            T = ep['q'].shape[0]
            sl = slice(at, at + T)
            self.readings[sl], self.raw[sl] = ep['readings_mm'], ep['raw_mm']
            self.cls[sl] = zone_class(ep['hit_class'])
            self.clear[sl] = np.clip(ep['clearance_m'], -CAP_M, CAP_M)
            self.q[sl], self.qd[sl] = ep['q'], ep['qd']
            self.alive[sl] = ep['sensor_alive'][None, :]
            at += T
            if log_every and (k + 1) % log_every == 0:
                print(f'  loaded {k + 1}/{len(indices)} episodes ({time.monotonic() - t0:.0f} s)', flush=True)
        if at != self.n:
            raise ValueError(f'manifest says {self.n} frames, episodes hold {at}')

    def batch(self, idx, torch):
        r = self.readings[idx].astype(np.float32)
        raw = self.raw[idx].astype(np.float32)
        valid = (r > 0).astype(np.float32)
        x = np.stack([valid, r / NORM_MM, raw / NORM_MM, (raw > 0).astype(np.float32),
                      np.broadcast_to(self.alive[idx][..., None], r.shape)], axis=2)  # (B, S, 5, 64)
        B, S = r.shape[:2]
        x = x.reshape(B, S, 5, 8, 8)
        joints = np.concatenate([self.q[idx], np.clip(self.qd[idx], -5, 5) / 5.0], axis=1)
        t = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(dt)  # noqa: E731
        return t(x), t(joints), t(self.cls[idx], torch.int64), t(self.clear[idx])


# -- the model -------------------------------------------------------------------

def build_model(torch, n_sensors, n_joints, n_skins, width=32, emb=16):
    nn = torch.nn

    class SkinNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(5, width, 3, padding=1), nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
            self.sensor = nn.Embedding(n_sensors, emb)
            self.joints = nn.Sequential(nn.Linear(2 * n_joints, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            self.zone = nn.Sequential(nn.Conv2d(width + emb + 64, 64, 1), nn.ReLU(), nn.Conv2d(64, 3, 1))
            self.clear = nn.Sequential(nn.Linear(n_sensors * 2 * width + 64, 256), nn.ReLU(),
                                       nn.Linear(256, n_skins))
            self.register_buffer('ids', torch.arange(n_sensors))

        def forward(self, x, joints):
            B, S = x.shape[:2]
            f = self.enc(x.reshape(B * S, 5, 8, 8))                          # (B*S, W, 8, 8)
            j = self.joints(joints)                                          # (B, 64)
            e = self.sensor(self.ids).repeat(B, 1)                           # (B*S, emb)
            jj = j.repeat_interleave(S, 0)
            ctx = torch.cat([e, jj], 1)[:, :, None, None].expand(-1, -1, 8, 8)
            logits = self.zone(torch.cat([f, ctx], 1))                       # (B*S, 3, 8, 8)
            logits = logits.reshape(B, S, 3, 64)
            pooled = torch.cat([f.amax((2, 3)), f.mean((2, 3))], 1).reshape(B, -1)
            clear = CAP_M * torch.tanh(self.clear(torch.cat([pooled, j], 1)))  # (B, K), m
            return logits, clear

    return SkinNet()


# -- metrics -----------------------------------------------------------------------

def zone_loss(ce, logits, y):
    """Cross-entropy of zone logits (B, S, 3, 64) against classes (B, S, 64): one
    64-zone sample per sensor, classes on dim 1 as CrossEntropyLoss expects."""
    return ce(logits.reshape(-1, 3, 64), y.reshape(-1, 64))


def zone_report(pred, true, valid):
    """Accuracy, and precision/recall of self and env, over zones with a valid reading."""
    p, t = pred[valid], true[valid]
    out = {'accuracy_valid_zones': float((p == t).mean()), 'valid_zones': int(valid.sum())}
    for c in (SELF, ENV):
        tp = int(((p == c) & (t == c)).sum())
        out[f'{CLASS_NAMES[c]}_precision'] = tp / max(int((p == c).sum()), 1)
        out[f'{CLASS_NAMES[c]}_recall'] = tp / max(int((t == c).sum()), 1)
    return out


def clearance_report(pred, true):
    near_t, near_p = true < NEAR_M, pred < NEAR_M
    tp = int((near_t & near_p).sum())
    return {'mae_m': float(np.abs(pred - true).mean()),
            'mae_m_when_true_under_0.3m': float(np.abs(pred - true)[true < 0.3].mean()) if (true < 0.3).any() else None,
            f'near_{NEAR_M:g}m_recall': tp / max(int(near_t.sum()), 1),
            f'near_{NEAR_M:g}m_precision': tp / max(int(near_p.sum()), 1),
            'near_frames': int(near_t.sum())}


def nearest_reading_clearance(frames, skin_of_sensor, n_skins, drop_self):
    """Per frame and skin: the nearest valid reading of that skin's sensors (m), capped."""
    r = frames.readings.astype(np.float32) / 1000.0
    ok = frames.readings > 0
    if drop_self:
        ok &= frames.cls != SELF
    r = np.where(ok, r, np.inf).min(axis=2)                                # (N, S)
    out = np.full((frames.n, n_skins), CAP_M, np.float32)
    for k in range(n_skins):
        out[:, k] = np.minimum(r[:, skin_of_sensor == k].min(axis=1), CAP_M)
    return out


def split(ds):
    n = len(ds)
    n_test = max(1, n // 10)
    n_val = max(1, n // 10)
    idx = list(range(n))
    return idx[:n - n_test - n_val], idx[n - n_test - n_val:n - n_test], idx[n - n_test:]


def skin_of_sensors(ds):
    names = ds.info['sensors']
    skins = ds.info['skins']
    return np.array([next(k for k, s in enumerate(skins) if n.startswith(s + '_sensor_')) for n in names])


# -- train / eval --------------------------------------------------------------------

def predict(torch, model, frames, batch=512):
    model.eval()
    zones, clears = [], []
    with torch.no_grad():
        for s in range(0, frames.n, batch):
            idx = np.arange(s, min(s + batch, frames.n))
            x, j, _, _ = frames.batch(idx, torch)
            logits, clear = model(x, j)
            zones.append(logits.argmax(2).numpy().astype(np.uint8))
            clears.append(clear.numpy())
    return np.concatenate(zones), np.concatenate(clears)


def evaluate(torch, model, frames, skin_of_sensor, n_skins, threshold_mm, sphere=None):
    zp, cp = predict(torch, model, frames)
    valid = frames.readings > 0
    all_env = np.where(valid, ENV, NONE).astype(np.uint8)
    thresh = np.where(valid, np.where(frames.readings < threshold_mm, SELF, ENV), NONE).astype(np.uint8)
    true_c = np.clip(frames.clear, 0.0, CAP_M)
    return {
        'zones': {'model': zone_report(zp, frames.cls, valid),
                  'all_env': zone_report(all_env, frames.cls, valid),
                  f'self_threshold_{threshold_mm}mm': zone_report(thresh, frames.cls, valid),
                  **({f'sphere_filter_{int(round(sphere[0] * 1000))}mm_margin': zone_report(sphere[1], frames.cls, valid)}
                     if sphere is not None else {})},
        'clearance': {'model': clearance_report(np.clip(cp, 0, CAP_M), true_c),
                      'nearest_reading': clearance_report(
                          nearest_reading_clearance(frames, skin_of_sensor, n_skins, False), true_c),
                      'nearest_reading_oracle_self_filter': clearance_report(
                          nearest_reading_clearance(frames, skin_of_sensor, n_skins, True), true_c)},
    }


SPHERE_MARGINS_M = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)


def _quat_to_mat(qwxyz):
    w, x, y, z = (qwxyz[..., i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def sphere_zone_predictions(ds, indices, margins=SPHERE_MARGINS_M, mjcf=None, extra=()):
    """{margin: (N, S, 64) zone classes} from the sphere self-filter, in Frames order."""
    from . import vl53l5cx
    from .generate import DEFAULT_MJCF
    from .sphere_filter import SphereSelfFilter
    f = SphereSelfFilter(mjcf or DEFAULT_MJCF, extra=extra)
    out = {m: [] for m in margins}
    for i in indices:
        ep = ds.episode_arrays(i, ('readings_mm', 'sensor_pos', 'sensor_quat', 'q', 'joint_names'))
        r = ep['readings_mm']
        T, S = r.shape[:2]
        R = _quat_to_mat(ep['sensor_quat'].astype(np.float64))                 # (T, S, 3, 3)
        local = vl53l5cx.ZONE_RAYS[None, None] * (r[..., None] / 1000.0)        # (T, S, 64, 3)
        pts = ep['sensor_pos'][:, :, None, :] + np.einsum('tsij,tszj->tszi', R, local)
        valid = r > 0
        sd = np.full((T, S, 64), np.inf)
        names = [str(n) for n in ep['joint_names']]
        for t in range(T):
            f.set_joints(names, ep['q'][t])
            v = valid[t]
            if v.any():
                sd[t][v] = f.clearance(pts[t][v])
        for m in margins:
            out[m].append(np.where(valid, np.where(sd <= m, SELF, ENV), NONE).astype(np.uint8))
    return {m: np.concatenate(v) for m, v in out.items()}


def best_sphere_margin(ds, indices, frames, mjcf=None, extra=()):
    preds = sphere_zone_predictions(ds, indices, mjcf=mjcf, extra=extra)
    valid = frames.readings > 0
    scores = {m: float((p[valid] == frames.cls[valid]).mean()) for m, p in preds.items()}
    return max(scores, key=scores.get), scores


def best_threshold(frames):
    """The 'valid returns under X mm are self' threshold with the best accuracy."""
    valid = frames.readings > 0
    r, t = frames.readings[valid], frames.cls[valid]
    best = (0.0, 0)
    for x in range(20, 801, 20):
        acc = float((np.where(r < x, SELF, ENV) == t).mean())
        best = max(best, (acc, x))
    return best[1]


def train(args):
    import torch
    torch.manual_seed(args.seed)
    ds = Dataset(args.data)
    tr, va, te = split(ds)
    print(f'{len(ds)} episodes: train {len(tr)}, val {len(va)}, test {len(te)}', flush=True)
    t0 = time.monotonic()
    ftr, fva = Frames(ds, tr), Frames(ds, va)
    print(f'loaded {ftr.n} train / {fva.n} val frames in {time.monotonic() - t0:.0f} s', flush=True)
    S, J, K = ftr.readings.shape[1], ftr.q.shape[1], ftr.clear.shape[1]
    model = build_model(torch, S, J, K, width=args.width)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(ftr.n / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    # Class weights: self is rare and matters most.
    counts = np.bincount(ftr.cls.ravel(), minlength=3).astype(float)
    w = torch.tensor((counts.sum() / (3 * np.maximum(counts, 1))) ** 0.5, dtype=torch.float32)
    ce = torch.nn.CrossEntropyLoss(weight=w)
    rng = np.random.default_rng(args.seed)
    log = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(ftr.n)
        tl, n = 0.0, 0
        te0 = time.monotonic()
        for s in range(0, ftr.n, args.batch):
            idx = np.sort(order[s:s + args.batch])
            x, j, y, c = ftr.batch(idx, torch)
            logits, clear = model(x, j)
            loss = zone_loss(ce, logits, y) + \
                args.clear_weight * torch.nn.functional.smooth_l1_loss(clear, c.clamp(min=0.0), beta=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss.detach()) * len(idx)
            n += len(idx)
        zp, cp = predict(torch, model, fva)
        vz = zone_report(zp, fva.cls, fva.readings > 0)
        vc = clearance_report(np.clip(cp, 0, CAP_M), np.clip(fva.clear, 0, CAP_M))
        log.append({'epoch': epoch + 1, 'train_loss': tl / n, 'val_zone_acc': vz['accuracy_valid_zones'],
                    'val_self_recall': vz['self_recall'], 'val_clear_mae': vc['mae_m'],
                    'val_near_recall': vc[f'near_{NEAR_M:g}m_recall'], 'seconds': time.monotonic() - te0})
        print(f"epoch {epoch + 1}/{args.epochs}: loss {tl / n:.4f} | val zone acc {vz['accuracy_valid_zones']:.4f}, "
              f"self P/R {vz['self_precision']:.3f}/{vz['self_recall']:.3f} | clearance MAE "
              f"{vc['mae_m'] * 100:.1f} cm, near<10cm R {vc[f'near_{NEAR_M:g}m_recall']:.3f} "
              f"| {time.monotonic() - te0:.0f} s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    thr = best_threshold(fva)
    torch.save(model.state_dict(), os.path.join(args.out, 'model.pt'))
    cfg = {'n_sensors': S, 'n_joints': J, 'n_skins': K, 'width': args.width, 'self_threshold_mm': thr,
           'sensors': ds.info['sensors'], 'skins': ds.info['skins'], 'joints': ds.info['joints'],
           'data': os.path.abspath(args.data), 'split': {'train': len(tr), 'val': len(va), 'test': len(te)},
           'train_log': log, 'args': vars(args)}
    with open(os.path.join(args.out, 'config.json'), 'w', newline='\n') as f:
        json.dump(cfg, f, indent=1)
    del ftr
    evaluate_cli(args.data, args.out)


def evaluate_cli(data, model_dir):
    import torch
    cfg = json.load(open(os.path.join(model_dir, 'config.json'), encoding='utf-8'))
    ds = Dataset(data)
    _, _, te = split(ds)
    fte = Frames(ds, te)
    model = build_model(torch, cfg['n_sensors'], cfg['n_joints'], cfg['n_skins'], cfg['width'])
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt'), weights_only=True))
    tr, va, _ = split(ds)
    mjcf = ds.info.get('mjcf')                   # the MJCF the episodes were generated on
    margin, scores = best_sphere_margin(ds, va, Frames(ds, va, log_every=0), mjcf)
    print(f"sphere filter: validation accuracy by margin {({f'{m * 1000:.0f}mm': round(v, 4) for m, v in scores.items()})}"
          f" -> {margin * 1000:.0f} mm", flush=True)
    sphere = (margin, sphere_zone_predictions(ds, te, (margin,), mjcf)[margin])
    rep = evaluate(torch, model, fte, skin_of_sensors(ds), cfg['n_skins'], cfg['self_threshold_mm'], sphere)
    rep['sphere_margin_val_accuracy'] = {str(m): v for m, v in scores.items()}
    rep['test_frames'] = fte.n
    rep['test_episodes'] = len(te)
    with open(os.path.join(model_dir, 'test_report.json'), 'w', newline='\n') as f:
        json.dump(rep, f, indent=1)
    print(f'\ntest: {len(te)} held-out episodes, {fte.n} frames')
    print('zones with a valid reading:      accuracy   self P / R      env P / R')
    for k, v in rep['zones'].items():
        print(f"  {k:30s} {v['accuracy_valid_zones']:.4f}   {v['self_precision']:.3f} / {v['self_recall']:.3f}"
              f"   {v['env_precision']:.3f} / {v['env_recall']:.3f}")
    print('clearance per skin:              MAE      MAE(<0.3m)   near<10cm R / P')
    for k, v in rep['clearance'].items():
        m3 = v['mae_m_when_true_under_0.3m']
        print(f"  {k:30s} {v['mae_m'] * 100:5.1f} cm  {m3 * 100 if m3 is not None else float('nan'):6.1f} cm"
              f"    {v[f'near_{NEAR_M:g}m_recall']:.3f} / {v[f'near_{NEAR_M:g}m_precision']:.3f}")
    print(f"  (near frames in test: {rep['clearance']['model']['near_frames']})")
    return rep


def baselines_cli(data):
    """The no-learning self-filters on the test split: all-env, the distance threshold,
    and the sphere filter with GOLEM's spheres alone and with the fitted leg spheres."""
    from .sphere_filter import LEG_SPHERES
    ds = Dataset(data)
    _, va, te = split(ds)
    fva, fte = Frames(ds, va, log_every=0), Frames(ds, te, log_every=0)
    mjcf = ds.info.get('mjcf')
    valid = fte.readings > 0
    thr = best_threshold(fva)
    rows = {'all_env': np.where(valid, ENV, NONE).astype(np.uint8),
            f'self_threshold_{thr}mm': np.where(valid, np.where(fte.readings < thr, SELF, ENV), NONE).astype(np.uint8)}
    for name, extra in (('golem_spheres', ()), ('golem_plus_leg_spheres', (LEG_SPHERES,))):
        m, _ = best_sphere_margin(ds, va, fva, mjcf, extra)
        rows[f'{name}_{m * 1000:.0f}mm'] = sphere_zone_predictions(ds, te, (m,), mjcf, extra)[m]
    rep = {k: zone_report(v, fte.cls, valid) for k, v in rows.items()}
    self_share = float((fte.cls[valid] == SELF).mean())
    print(f'{data}: {len(te)} test episodes, skins {ds.info["skins"]}; '
          f'{self_share * 100:.1f}% of valid zones are self-returns')
    print('zones with a valid reading:      accuracy   self P / R      env P / R')
    for k, v in rep.items():
        print(f"  {k:30s} {v['accuracy_valid_zones']:.4f}   {v['self_precision']:.3f} / {v['self_recall']:.3f}"
              f"   {v['env_precision']:.3f} / {v['env_recall']:.3f}")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('data')
    t.add_argument('--out', required=True)
    t.add_argument('--epochs', type=int, default=6)
    t.add_argument('--batch', type=int, default=256)
    t.add_argument('--lr', type=float, default=2e-3)
    t.add_argument('--width', type=int, default=32)
    t.add_argument('--clear-weight', type=float, default=5.0)
    t.add_argument('--seed', type=int, default=0)
    e = sub.add_parser('eval')
    e.add_argument('data')
    e.add_argument('--model', required=True)
    b = sub.add_parser('baselines', help='score only the no-learning self-filters')
    b.add_argument('data')
    args = ap.parse_args(argv)
    if args.cmd == 'train':
        train(args)
    elif args.cmd == 'baselines':
        baselines_cli(args.data)
    else:
        evaluate_cli(args.data, args.model)


if __name__ == '__main__':
    main()
