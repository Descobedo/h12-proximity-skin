"""Receive HIRO skin boards' VL53L5CX frames on this PC: status, identify, record, twin.

    python3 -m golem_proximity_skin.hiro_udp status                    # rates, nearest, per sensor
    python3 -m golem_proximity_skin.hiro_udp identify                  # which firmware sensor is covered
    python3 -m golem_proximity_skin.hiro_udp record run1.npz --duration 60
    python3 -m golem_proximity_skin.hiro_udp twin                      # live readings on the sim H1-2
    python3 -m golem_proximity_skin.hiro_udp mask open.npz --write     # zones the skin obscures

HIRO's firmware (proximity_point_motion, Arduino/) sends each sensor's frame as
one UDP datagram to multicast 239.0.0.1 on its skin's port:
  [1 byte firmware sensor index][64 x uint16 filtered mm][64 x uint16 raw mm]   little-endian
"filtered" puts 4000 where the sensor flagged the zone invalid (target_status
255); older sketches send only the first 129 bytes. Here a zone at 4000 mm or
more is vl53l5cx.NO_TARGET (0), as everywhere in this package.

Which skins to listen for, and which layout sensor each firmware index is, come
from the board map (config/hiro_boards.yaml): a board may drive fewer sensors
than its skin has, in its own order, and lists per sensor the zones its skin
obscures (masked_zones): those are reported as NO_TARGET, raw 0, wherever
this module hands frames on. Ports come from the layout
(config/h12_skin.yaml, HIRO's sensors_config.yaml). No ROS needed.
"""
import argparse
import os
import select
import socket
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import yaml

from . import vl53l5cx
from .layout import load_layout

GROUP = '239.0.0.1'
FIRMWARE_NO_TARGET_MM = 4000
ZONES = vl53l5cx.NUM_ZONES
SHORT_PACKET = 1 + 2 * ZONES          # [id][filtered]
FULL_PACKET = 1 + 4 * ZONES           # [id][filtered][raw]

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LAYOUT = os.path.join(HERE, '..', 'config', 'h12_skin.yaml')
DEFAULT_BOARDS = os.path.join(HERE, '..', 'config', 'hiro_boards.yaml')


# -- the wire format ------------------------------------------------------------

def parse_packet(data: bytes):
    """(firmware index, readings mm (64,) uint16 with NO_TARGET for no return,
    raw mm (64,) or None); None for a datagram too short to be a frame."""
    if len(data) < SHORT_PACKET:
        return None
    filt = np.frombuffer(data, dtype='<u2', count=ZONES, offset=1)
    raw = (np.frombuffer(data, dtype='<u2', count=ZONES, offset=SHORT_PACKET).copy()
           if len(data) >= FULL_PACKET else None)
    readings = np.where(filt >= FIRMWARE_NO_TARGET_MM, vl53l5cx.NO_TARGET, filt).astype(np.uint16)
    return int(data[0]), readings, raw


def encode_packet(index: int, readings_mm, raw_mm=None) -> bytes:
    """A frame as the firmware sends it (readings in this package's convention)."""
    r = np.asarray(readings_mm, dtype=np.uint16).reshape(ZONES)
    filt = np.where(r == vl53l5cx.NO_TARGET, FIRMWARE_NO_TARGET_MM, r).astype('<u2')
    raw = filt if raw_mm is None else np.asarray(raw_mm, dtype='<u2').reshape(ZONES)
    return bytes([index]) + filt.tobytes() + raw.tobytes()


# -- the board map ---------------------------------------------------------------

@dataclass
class Board:
    skin: str
    port: int
    sensors: List[int]                      # firmware index k -> layout sensor sensors[k]
    verified: bool = False
    # firmware index -> zones to ignore (the skin obscures them); see `hiro_udp mask`
    masked_zones: Dict[int, List[int]] = field(default_factory=dict)

    def layout_sensor(self, k: int) -> int:
        return self.sensors[k] if 0 <= k < len(self.sensors) else -1

    def zone_mask(self, k: int) -> np.ndarray:
        """(64,) bool: True for the zones of firmware sensor k to ignore."""
        m = np.zeros(ZONES, dtype=bool)
        m[self.masked_zones.get(k, [])] = True
        return m


def load_boards(path: str, layout) -> List[Board]:
    doc = yaml.safe_load(open(path, encoding='utf-8')) or {}
    skins = {s.name: s for s in layout.skins}
    boards = []
    for b in doc.get('boards', []):
        skin = skins.get(b['skin'])
        if skin is None:
            raise ValueError(f"{path}: no skin {b['skin']!r} in the layout")
        port = int(b.get('port', skin.port or 0))
        if not port:
            raise ValueError(f"{path}: skin {skin.name} has no port")
        sensors = [int(i) for i in b.get('sensors', range(len(skin.sensors)))]
        bad = [i for i in sensors if not 0 <= i < len(skin.sensors)]
        if bad or len(set(sensors)) != len(sensors):
            raise ValueError(f'{path}: {skin.name} sensors {sensors} must be distinct, '
                             f'0..{len(skin.sensors) - 1}')
        masked = {int(k): sorted({int(z) for z in v}) for k, v in (b.get('masked_zones') or {}).items()}
        for k, zones in masked.items():
            if not 0 <= k < len(sensors) or any(not 0 <= z < ZONES for z in zones):
                raise ValueError(f'{path}: {skin.name} masked_zones {k}: {zones} must name firmware '
                                 f'0..{len(sensors) - 1} and zones 0..{ZONES - 1}')
        boards.append(Board(skin.name, port, sensors, bool(b.get('verified', False)), masked))
    if len({b.port for b in boards}) != len(boards):
        raise ValueError(f'{path}: two boards on one port')
    return boards


# -- receiving -----------------------------------------------------------------

@dataclass
class Frame:
    t_ns: int                 # wall clock at receipt (time.time_ns)
    board: Board
    index: int                # firmware sensor index
    readings: np.ndarray      # (64,) uint16 mm, NO_TARGET = no return or masked
    raw: Optional[np.ndarray]  # (64,) raw mm, 0 where masked; None from short packets
    packet: bytes             # as received, masked zones included
    source: str
    masked: np.ndarray = None  # (64,) bool, the board's masked zones for this sensor


def local_ip_toward(host: str = '192.168.123.1') -> str:
    """This PC's address on the route toward `host` (nothing is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((host, 9))
        return s.getsockname()[0]
    finally:
        s.close()


class Receiver:
    """One UDP socket per board; poll() returns the frames that have arrived."""

    def __init__(self, boards: List[Board], iface: str = '', multicast: bool = True):
        self.boards = {b.port: b for b in boards}
        self.iface = iface or local_ip_toward()
        self.socks = {}
        for port in self.boards:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            s.bind(('0.0.0.0', port))
            if multicast:
                s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                             socket.inet_aton(GROUP) + socket.inet_aton(self.iface))
            s.setblocking(False)
            self.socks[s] = port
        self.sources: Dict[int, set] = {p: set() for p in self.boards}
        self.dropped = 0

    def poll(self, timeout: float = 0.05) -> List[Frame]:
        out = []
        ready, _, _ = select.select(list(self.socks), [], [], timeout)
        for s in ready:
            port = self.socks[s]
            board = self.boards[port]
            while True:
                try:
                    data, addr = s.recvfrom(4096)
                except (BlockingIOError, ConnectionResetError):
                    break
                p = parse_packet(data)
                if p is None:
                    self.dropped += 1
                    continue
                if addr[0] not in self.sources[port]:
                    self.sources[port].add(addr[0])
                    print(f'[hiro_udp] board {addr[0]} -> port {port} ({board.skin}), '
                          f'{len(data)}-byte frames', flush=True)
                k, readings, raw = p
                masked = board.zone_mask(k)
                if masked.any():          # the skin obscures these zones: never report them
                    readings = np.where(masked, vl53l5cx.NO_TARGET, readings).astype(np.uint16)
                    if raw is not None:
                        raw = np.where(masked, 0, raw).astype(np.uint16)
                out.append(Frame(time.time_ns(), board, k, readings, raw, data, addr[0], masked))
        return out

    def close(self):
        for s in self.socks:
            s.close()


class Stats:
    """Per (port, firmware index): receive rate over a window, latest frame."""

    def __init__(self, window_s: float = 2.0):
        self.window = window_s
        self.times: Dict[tuple, List[float]] = {}
        self.latest: Dict[tuple, Frame] = {}

    def add(self, f: Frame):
        k = (f.board.port, f.index)
        now = time.monotonic()
        ts = [t for t in self.times.get(k, []) if now - t < self.window]
        ts.append(now)
        self.times[k] = ts
        self.latest[k] = f

    def rate(self, k) -> float:
        now = time.monotonic()
        return sum(1 for t in self.times.get(k, []) if now - t < self.window) / self.window

    def age(self, k) -> float:
        ts = self.times.get(k)
        return time.monotonic() - ts[-1] if ts else float('inf')


def nearest_mm(readings) -> int:
    v = readings[readings != vl53l5cx.NO_TARGET]
    return int(v.min()) if v.size else 0


# -- recording -----------------------------------------------------------------

class Recorder:
    """Frames -> one .npz. The arrays HIRO's recorder writes (timestamps, packets,
    sensor_ids, skins, skin_indices) plus what this package needs to place them."""

    def __init__(self, path: str, boards: List[Board], layout_path: str, boards_path: str):
        self.path = path
        self.boards = boards
        self.offsets, n = {}, 0
        for b in boards:            # global sensor id: firmware index + the board's offset
            self.offsets[b.port] = n
            n += len(b.sensors)
        self.layout_yaml = open(layout_path, encoding='utf-8').read()
        self.boards_yaml = open(boards_path, encoding='utf-8').read()
        self.rows = []

    def add(self, f: Frame):
        self.rows.append(f)

    def save(self) -> str:
        n = len(self.rows)
        packets = np.zeros((n, FULL_PACKET), dtype=np.uint8)
        lengths = np.zeros(n, dtype=np.uint16)
        for i, f in enumerate(self.rows):
            b = np.frombuffer(f.packet[:FULL_PACKET], dtype=np.uint8)
            packets[i, :len(b)] = b
            lengths[i] = len(f.packet)
        np.savez_compressed(
            self.path,
            timestamps=np.array([f.t_ns for f in self.rows], dtype=np.uint64),
            packets=packets,
            packet_lengths=lengths,
            sensor_ids=np.array([self.offsets[f.board.port] + f.index for f in self.rows], dtype=np.uint8),
            skins=np.array([f.board.skin for f in self.rows], dtype='U32'),
            skin_indices=np.array([f.index for f in self.rows], dtype=np.uint8),
            layout_sensors=np.array([f.board.layout_sensor(f.index) for f in self.rows], dtype=np.int16),
            ports=np.array([f.board.port for f in self.rows], dtype=np.uint16),
            sources=np.array([f.source for f in self.rows], dtype='U15'),
            readings_mm=(np.stack([f.readings for f in self.rows]) if n
                         else np.zeros((0, ZONES), dtype=np.uint16)),
            masked_zones=(np.stack([f.masked if f.masked is not None else np.zeros(ZONES, bool)
                                    for f in self.rows]) if n else np.zeros((0, ZONES), dtype=bool)),
            layout_yaml=np.array(self.layout_yaml),
            boards_yaml=np.array(self.boards_yaml),
            no_target_mm=np.array(vl53l5cx.NO_TARGET),
        )
        return self.path


# -- masks ---------------------------------------------------------------------

def measure_masks(path, near_mm=60, open_mm=800, min_share=0.5, margin=1):
    """{(port, firmware index): (zones or None, stats)}: zones the skin obscures, from a
    recording made with the skin in the open.

    Only frames where the sensor mostly sees the room (median zone beyond open_mm) count;
    a zone reading under near_mm in at least min_share of them is obscured. The mask then
    grows by `margin` zones (4-neighbour): zones beside the shell's edge mix the shell with
    the scene and read short without reading near.
    """
    d = np.load(path, allow_pickle=False)
    raw = d['packets'][:, SHORT_PACKET:FULL_PACKET].copy().view('<u2').astype(float)
    out = {}
    for port in np.unique(d['ports']):
        for k in np.unique(d['skin_indices'][d['ports'] == port]):
            sel = (d['ports'] == port) & (d['skin_indices'] == k)
            R = np.where((raw[sel] > 0) & (raw[sel] < FIRMWARE_NO_TARGET_MM), raw[sel], np.nan)
            with np.errstate(all='ignore'):
                open_ = np.nanmedian(R, axis=1) > open_mm
            if open_.sum() < 20:
                out[(int(port), int(k))] = (None, {'open_frames': int(open_.sum())})
                continue
            share = (R[open_] < near_mm).mean(axis=0)
            core = (share >= min_share).reshape(8, 8)
            grown = core.copy()
            for _ in range(margin):
                g = grown.copy()
                g[1:] |= grown[:-1]
                g[:-1] |= grown[1:]
                g[:, 1:] |= grown[:, :-1]
                g[:, :-1] |= grown[:, 1:]
                grown = g
            out[(int(port), int(k))] = (np.nonzero(grown.ravel())[0].tolist(),
                                        {'open_frames': int(open_.sum()), 'obscured': int(core.sum()),
                                         'with_margin': int(grown.sum())})
    return out


def write_masks(boards_path, skin, masks):
    """Set `masked_zones` of the board for `skin` in the board map, keeping its comments."""
    lines = open(boards_path, encoding='utf-8').read().split('\n')
    start = next((i for i, l in enumerate(lines) if l.strip() == f'- skin: {skin}'), None)
    if start is None:
        raise ValueError(f'{boards_path}: no board for {skin}')
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].strip().startswith('- skin:') or (lines[i] and not lines[i].startswith(' '))),
               len(lines))
    block = [l for l in lines[start:end] if not l.strip().startswith('masked_zones:')]
    indent = block[0][:len(block[0]) - len(block[0].lstrip())] + '  '
    value = '{' + ', '.join(f'{k}: [{", ".join(map(str, z))}]' for k, z in sorted(masks.items()) if z) + '}'
    at = next((i for i, l in enumerate(block) if l.strip().startswith('sensors:')), 0) + 1
    block.insert(at, f'{indent}masked_zones: {value}')
    with open(boards_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines[:start] + block + lines[end:]))


def run_mask(recording, boards, boards_path, write, margin):
    found = measure_masks(recording, margin=margin)
    for b in boards:
        masks = {}
        for k in range(len(b.sensors)):
            zones, st = found.get((b.port, k), (None, {'open_frames': 0}))
            if zones is None:
                print(f'[mask] {b.skin} firmware {k}: only {st["open_frames"]} frames facing the open '
                      'room; not measured', flush=True)
                continue
            masks[k] = zones
            grid = np.zeros(ZONES, bool)
            grid[zones] = True
            print(f'[mask] {b.skin} firmware {k} ({b.skin}_sensor_{b.layout_sensor(k)}): '
                  f'{st["obscured"]} obscured, {st["with_margin"]} masked with a {margin}-zone margin '
                  f'({st["open_frames"]} open frames)', flush=True)
            if grid.any():
                print('    ' + '\n    '.join(''.join('#' if v else '.' for v in row) for row in grid.reshape(8, 8)))
        if write and masks:
            write_masks(boards_path, b.skin, masks)
            print(f'[mask] wrote {b.skin} masked_zones to {boards_path}', flush=True)


# -- modes ---------------------------------------------------------------------

def _status_line(stats, boards):
    parts = []
    for b in boards:
        for k in range(len(b.sensors)):
            key = (b.port, k)
            if key not in stats.latest:
                parts.append(f'{b.skin}[{k}] --')
                continue
            f = stats.latest[key]
            stale = ' STALE' if stats.age(key) > 1.0 else ''
            near = nearest_mm(f.readings)
            parts.append(f'{b.skin}[{k}] {stats.rate(key):4.1f}Hz '
                         f'{(str(near) + "mm") if near else "none":>6}{stale}')
    return ' | '.join(parts)


def run_status(rx, boards, duration, every=2.0):
    stats, t0, last = Stats(), time.monotonic(), 0.0
    while duration <= 0 or time.monotonic() - t0 < duration:
        for f in rx.poll():
            stats.add(f)
        if time.monotonic() - last >= every:
            last = time.monotonic()
            print(_status_line(stats, boards), flush=True)
    return stats


def run_identify(rx, boards, duration, cover_mm=60, clear_mm=150):
    """Report which firmware sensor is covered: its nearest zone under cover_mm
    while every other sensor of its board reads beyond clear_mm (or nothing)."""
    print(f'[identify] cover one sensor at a time (within {cover_mm} mm); '
          'Ctrl+C to stop', flush=True)
    stats, t0, last_seen = Stats(), time.monotonic(), {}
    while duration <= 0 or time.monotonic() - t0 < duration:
        for f in rx.poll():
            stats.add(f)
        for b in boards:
            near = {k: nearest_mm(stats.latest[(b.port, k)].readings)
                    for k in range(len(b.sensors)) if (b.port, k) in stats.latest
                    and stats.age((b.port, k)) < 1.0}
            covered = [k for k, d in near.items() if d and d < cover_mm]
            clear = all((not d) or d > clear_mm for k, d in near.items() if k not in covered)
            hit = covered[0] if len(covered) == 1 and clear else None
            if hit is not None and last_seen.get(b.port) != hit:
                print(f'[identify] {b.skin}: firmware sensor {hit} covered '
                      f'({near[hit]} mm); board map says layout sensor {b.layout_sensor(hit)}'
                      f' -> frame {b.skin}_sensor_{b.layout_sensor(hit)}', flush=True)
            last_seen[b.port] = hit


def run_record(rx, boards, out, duration, layout_path, boards_path):
    rec = Recorder(out, boards, layout_path, boards_path)
    stats, t0, last = Stats(), time.monotonic(), 0.0
    print(f'[record] -> {out}' + (f' for {duration:g} s' if duration > 0 else ', Ctrl+C to stop'), flush=True)
    stop = os.environ.get('HIRO_UDP_STOP_FILE', '')    # stop and save once this file exists
    try:
        while duration <= 0 or time.monotonic() - t0 < duration:
            if stop and os.path.exists(stop):
                print(f'[record] stop file {stop} found', flush=True)
                break
            for f in rx.poll():
                stats.add(f)
                rec.add(f)
            if time.monotonic() - last >= 2.0:
                last = time.monotonic()
                print(f'[record] {len(rec.rows)} frames | ' + _status_line(stats, boards), flush=True)
    except KeyboardInterrupt:
        pass
    path = rec.save()
    span = (rec.rows[-1].t_ns - rec.rows[0].t_ns) / 1e9 if len(rec.rows) > 1 else 0.0
    print(f'[record] wrote {path}: {len(rec.rows)} frames over {span:.1f} s', flush=True)
    return path


def run_twin(rx, boards, layout, mjcf, pose, fov_draw):
    """The sim H1-2 with the board map's skins; each mapped sensor draws its real
    zones (dots) and field of view, at the robot's pose (`pose`, joint=rad)."""
    import mujoco
    from mujoco import viewer as mj_viewer

    from .layout import SkinLayout
    from .mujoco_tof import ToFSkinCaster
    from .preview import add_dot, add_fov, build_spec, sensor_hues

    names = [b.skin for b in boards]
    sub = SkinLayout(skins=[s for s in layout.skins if s.name in names], rate_hz=layout.rate_hz)
    for s in sub.skins:
        s.enabled = True
    model = build_spec(mjcf, sub).compile()
    data = mujoco.MjData(model)
    for name, q in pose.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise KeyError(f'no joint {name!r}')
        data.qpos[model.jnt_qposadr[j]] = q
    mujoco.mj_forward(model, data)
    caster = ToFSkinCaster(model, sub)
    poses = {s.name: p for s, p in zip(caster.skins, caster.sensor_world_poses(data))}
    hues = sensor_hues(sum(len(b.sensors) for b in boards))
    stats = Stats()
    with mj_viewer.launch_passive(model, data) as v:
        first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sub.skins[0].parent)
        v.cam.lookat[:] = data.xpos[first]
        v.cam.distance = 0.9
        while v.is_running():
            for f in rx.poll(0.01):
                stats.add(f)
            with v.lock():
                v.user_scn.ngeom = 0
                h = 0
                for b in boards:
                    for k, ls in enumerate(b.sensors):
                        origin, R = poses[b.skin][ls]
                        hue = hues[h]
                        h += 1
                        f = stats.latest.get((b.port, k))
                        live = f is not None and stats.age((b.port, k)) < 1.0
                        if live:
                            pts, _ = vl53l5cx.readings_to_points(f.readings)
                            for p in pts:
                                add_dot(v.user_scn, origin + R @ p, 0.006, [*hue, 1.0])
                            near = nearest_mm(f.readings) / 1000.0
                        g = add_dot(v.user_scn, origin, 0.004, [*hue, 1.0])
                        if g is not None:
                            g.label = f'{b.skin}_sensor_{ls} (fw {k})'
                        depth = min(near, fov_draw) if live and near else 0.05
                        add_fov(v.user_scn, origin, R, depth,
                                [*hue, 0.9] if live else [0.5, 0.5, 0.5, 0.4], 1.5)
            v.sync()
            time.sleep(0.005)


def _pose(text):
    out = {}
    for item in filter(None, (t.strip() for t in (text or '').split(','))):
        name, val = item.split('=')
        out[name.strip()] = float(val)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mode', choices=('status', 'identify', 'record', 'twin', 'mask'))
    ap.add_argument('out', nargs='?', help='record: output .npz; mask: the recording to measure')
    ap.add_argument('--layout', default=DEFAULT_LAYOUT)
    ap.add_argument('--boards', default=DEFAULT_BOARDS, help='board map (firmware index -> layout sensor)')
    ap.add_argument('--skins', nargs='*', help='only these boards (by skin name)')
    ap.add_argument('--iface', default='', help="this PC's IP on the boards' network (default: auto)")
    ap.add_argument('--no-multicast', dest='multicast', action='store_false')
    ap.add_argument('--duration', type=float, default=0.0, help='s; 0 = until Ctrl+C')
    ap.add_argument('--mjcf', default=os.path.join(HERE, '..', '..', '..', '..', 'CL_Assets',
                                                   'mujoco_assets', 'scene_h1_2_magpie_pelvis_fixed.xml'))
    ap.add_argument('--pose', default='', help='twin: joint=rad,... (default: all zero)')
    ap.add_argument('--fov-draw', type=float, default=0.3, help='twin: longest FOV edge drawn (m)')
    ap.add_argument('--write', action='store_true', help='mask: write the measured masks to the board map')
    ap.add_argument('--margin', type=int, default=1, help='mask: zones added around each obscured zone')
    args = ap.parse_args(argv)
    if args.mode in ('record', 'mask') and not args.out:
        ap.error(f'{args.mode} needs a path')

    layout = load_layout(args.layout)
    boards = load_boards(args.boards, layout)
    if args.skins:
        boards = [b for b in boards if b.skin in args.skins]
    if not boards:
        ap.error('no boards selected')
    for b in boards:
        print(f'[hiro_udp] {b.skin}: port {b.port}, firmware 0..{len(b.sensors) - 1} -> layout sensors '
              f'{b.sensors}' + ('' if b.verified else ' (mapping NOT verified)')
              + ''.join(f'; firmware {k}: {len(z)} zones masked' for k, z in sorted(b.masked_zones.items())),
              flush=True)
    if args.mode == 'mask':
        return run_mask(args.out, boards, args.boards, args.write, args.margin)
    rx = Receiver(boards, args.iface, args.multicast)
    print(f'[hiro_udp] listening via {rx.iface}' + (f', multicast {GROUP}' if args.multicast else ''),
          flush=True)
    try:
        if args.mode == 'status':
            run_status(rx, boards, args.duration)
        elif args.mode == 'identify':
            run_identify(rx, boards, args.duration)
        elif args.mode == 'record':
            run_record(rx, boards, args.out, args.duration, args.layout, args.boards)
        else:
            run_twin(rx, boards, layout, args.mjcf, _pose(args.pose), args.fov_draw)
    except KeyboardInterrupt:
        pass
    finally:
        rx.close()


if __name__ == '__main__':
    sys.exit(main())
