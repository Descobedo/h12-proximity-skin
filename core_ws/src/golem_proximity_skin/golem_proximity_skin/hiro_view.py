"""Live 8x8 view of one HIRO skin board's VL53L5CX sensors, straight off its UDP stream.

    python3 -m golem_proximity_skin.hiro_view --port 5007            # shoulder_yaw_R (R2)
    python3 -m golem_proximity_skin.hiro_view --port 5012 --iface 192.168.123.5

HIRO's firmware (proximity_point_motion, Arduino/) sends each sensor's frame as
one UDP datagram to multicast 239.0.0.1 on its skin's port:
  [1 byte sensor index][64 x uint16 filtered mm][64 x uint16 raw mm]   (little-endian)
"filtered" puts 4000 where the sensor flagged the zone invalid (target_status
255); older sketches send only the first 129 bytes. Zones are shown in packet
order (zone k = row k // 8, column k % 8), not yet checked against the
sensor's physical orientation.

Each panel: distance (mm, near = red), its sensor's receive rate, nearest
valid zone, and zones without a target (grey). Needs numpy and matplotlib.
"""
import argparse
import select
import socket
import time

import numpy as np

GROUP = '239.0.0.1'
NO_TARGET_MM = 4000
ZONES = 64


def open_socket(port, iface, multicast=True):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    s.bind(('0.0.0.0', port))
    if multicast:
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                     socket.inet_aton(GROUP) + socket.inet_aton(iface))
    s.setblocking(False)
    return s


def parse(data):
    """(sensor index, filtered mm (64,), raw mm (64,) or None); None if malformed."""
    if len(data) < 1 + 2 * ZONES:
        return None
    filt = np.frombuffer(data, dtype='<u2', count=ZONES, offset=1)
    raw = (np.frombuffer(data, dtype='<u2', count=ZONES, offset=1 + 2 * ZONES)
           if len(data) >= 1 + 4 * ZONES else None)
    return data[0], filt, raw


def local_ip_toward(host='192.168.123.1'):
    """This PC's address on the route toward `host` (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((host, 9))
        return s.getsockname()[0]
    finally:
        s.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, default=5007, help="the skin's UDP port (HIRO sensors_config.yaml)")
    ap.add_argument('--sensors', type=int, default=0, help='panels to show (default: as many as arrive)')
    ap.add_argument('--iface', default='', help="this PC's IP on the skin's network (default: auto)")
    ap.add_argument('--no-multicast', dest='multicast', action='store_false')
    ap.add_argument('--max-mm', type=int, default=2000, help='colour scale top (mm)')
    args = ap.parse_args(argv)

    import matplotlib.pyplot as plt
    iface = args.iface or local_ip_toward()
    sock = open_socket(args.port, iface, args.multicast)
    print(f'listening on port {args.port}' + (f', multicast {GROUP} via {iface}' if args.multicast else ''), flush=True)

    frames, stamps, sources = {}, {}, set()
    plt.ion()
    fig, axes, images, labels = None, [], [], []

    def build(n):
        nonlocal fig, axes, images, labels
        if fig is not None:
            plt.close(fig)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, ax = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows), squeeze=False)
        fig.canvas.manager.set_window_title(f'HIRO skin, port {args.port}')
        cmap = plt.get_cmap('turbo_r').copy()
        cmap.set_over('0.35')
        axes, images, labels = [], [], []
        for k in range(rows * cols):
            a = ax[k // cols][k % cols]
            a.set_xticks([])
            a.set_yticks([])
            if k >= n:
                a.axis('off')
                continue
            im = a.imshow(np.full((8, 8), np.nan), cmap=cmap, vmin=0, vmax=args.max_mm)
            axes.append(a)
            images.append(im)
            labels.append(a.set_title(f'sensor {k}', fontsize=10))
        fig.colorbar(images[0], ax=ax, shrink=0.8, label='distance (mm); grey = no target',
                     extend='max')
        fig.show()

    shown = 0
    last_draw = 0.0
    try:
        while True:
            r, _, _ = select.select([sock], [], [], 0.02)
            if r:
                while True:
                    try:
                        data, addr = sock.recvfrom(4096)
                    except BlockingIOError:
                        break
                    p = parse(data)
                    if p is None:
                        continue
                    i, filt, _ = p
                    if addr[0] not in sources:
                        sources.add(addr[0])
                        print(f'board {addr[0]} -> port {args.port}, {len(data)}-byte packets', flush=True)
                    frames[i] = filt.copy()
                    stamps.setdefault(i, []).append(time.monotonic())
            want = args.sensors or (max(frames) + 1 if frames else 0)
            if want and want != shown:
                build(want)
                shown = want
            now = time.monotonic()
            if fig is None or now - last_draw < 1 / 30:
                plt.pause(0.001) if fig is not None else None
                continue
            last_draw = now
            if not plt.fignum_exists(fig.number):
                break
            for i in range(shown):
                if i not in frames:
                    labels[i].set_text(f'sensor {i}: no data')
                    continue
                a = frames[i].astype(float).reshape(8, 8)
                invalid = frames[i].reshape(8, 8) >= NO_TARGET_MM
                images[i].set_data(np.where(invalid, args.max_mm + 1, a))
                st = [t for t in stamps[i] if now - t < 2.0]
                stamps[i] = st
                hz = len(st) / 2.0
                valid = a[~invalid]
                near = f'{int(valid.min())} mm' if valid.size else 'none'
                stale = now - st[-1] > 1.0 if st else True
                labels[i].set_text(f'sensor {i}: {hz:4.1f} Hz' + (' (stale)' if stale else '')
                                   + f'\nnearest {near}, {int(invalid.sum())} no target')
            fig.canvas.draw_idle()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
