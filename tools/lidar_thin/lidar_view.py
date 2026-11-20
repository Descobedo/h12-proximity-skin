"""Live view of the H1-2 Livox cloud on any PC (Windows is fine), plus lag numbers.

Needs only numpy + matplotlib and plink (PuTTY) or ssh. The robot runs a
temporary, read-only lidar_tap.py (see robot_link.py) that thins each frame
before it crosses the Wi-Fi. No robot settings are touched.

    set H12_PW=...                    (or you are prompted)
    python lidar_view.py                          # 4000 pts/frame, all 30 Hz
    python lidar_view.py --radius 3 --pts 2000    # only the robot's surroundings
    python lidar_view.py --hz 10                  # fewer frames on a bad link

Status line, per 2 s window:
  robot   rate and delay (lidar stamp -> robot receive): is the robot keeping up?
  laptop  rate, link extra (robot send -> laptop, above the best seen), worst gap
  skipped frames the robot dropped because the link was still busy
Keys: f flip vertical, space pause. Every frame's timing -> lidar_view_timing.csv.
"""
import argparse, collections, os, sys, threading, time
import numpy as np
import robot_link

HERE = os.path.dirname(os.path.abspath(__file__))
HDR = __import__('struct').Struct('<4sIdddIII')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--range', type=float, default=8.0, help='half width of the view, m')
    robot_link.add_args(ap)
    a = ap.parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    proc = robot_link.start_tap(a, 'view')
    latest = [None]
    hist = collections.deque(maxlen=300)
    state = {'err': '', 'min_off': None, 'eof': False, 'robot': ''}
    log = open(os.path.join(HERE, 'lidar_view_timing.csv'), 'w')
    log.write('laptop_arrive,seq,stamp,robot_recv,robot_sent,points_total,points_sent,skipped\n')

    def reader():
        try:
            while True:
                magic, seq, stamp, recv, sent, n_tot, n, skipped = HDR.unpack(
                    robot_link.read_exact(proc.stdout, HDR.size))
                if magic != b'LDR1':
                    state['err'] = 'stream out of sync'; return
                xyz = np.frombuffer(robot_link.read_exact(proc.stdout, n * 6), '<f2')
                t = time.time(); off = t - sent
                state['min_off'] = off if state['min_off'] is None else min(state['min_off'], off)
                hist.append((t, stamp, recv, off, skipped))
                log.write(f'{t:.4f},{seq},{stamp:.4f},{recv:.4f},{sent:.4f},{n_tot},{n},{skipped}\n')
                latest[0] = xyz.reshape(n, 3).astype(np.float32)
        except EOFError:
            state['eof'] = True

    def errs():
        for line in proc.stderr:
            line = line.decode(errors='replace').strip()
            if line.startswith('tap:'):
                state['robot'] = line[5:]
            elif line:
                state['err'] = line[-140:]

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=errs, daemon=True).start()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.canvas.manager.set_window_title('H1-2 lidar (read-only, thinned on the robot)')
    R = a.range
    sc1 = ax1.scatter([], [], s=1.5, c=[], cmap='viridis', vmin=-1.5, vmax=1.5)
    sc2 = ax2.scatter([], [], s=1.5, c=[], cmap='viridis', vmin=-1.5, vmax=1.5)
    for ax, ttl, yl, ylim in ((ax1, 'top down', 'y (m)', (-R, R)), (ax2, 'side', 'z (m)', (-3, 3))):
        ax.set_xlim(-R, R); ax.set_ylim(*ylim); ax.set_aspect('equal')
        ax.set_title(ttl); ax.set_xlabel('x (m)'); ax.set_ylabel(yl)
        ax.plot(0, 0, 'r+', ms=12); ax.grid(alpha=0.3)
    txt = fig.text(0.01, 0.01, 'connecting...', family='monospace', fontsize=9, va='bottom')
    view = {'flip': 1.0, 'pause': False}

    def key(e):
        if e.key == 'f':
            view['flip'] *= -1
        elif e.key == ' ':
            view['pause'] = not view['pause']
    fig.canvas.mpl_connect('key_press_event', key)

    def stats():
        now = time.time()
        w = [h for h in hist if now - h[0] < 2.0]
        if len(w) < 3:
            return 'stream ended' if state['eof'] else 'waiting for frames...'
        t, stamp, recv, off, sk = (np.array(c) for c in zip(*w))
        d = (recv - stamp) * 1e3
        extra = (off - state['min_off']) * 1e3
        return (f'robot: {(len(t)-1)/max(stamp[-1]-stamp[0],1e-6):5.1f} Hz  delay {d.mean():4.0f} ms '
                f'(max {d.max():4.0f})  |  laptop: {(len(t)-1)/max(t[-1]-t[0],1e-6):5.1f} Hz  '
                f'link extra {extra.mean():4.0f} ms (max {extra.max():4.0f})  worst gap '
                f'{np.diff(t).max()*1e3:4.0f} ms  skipped {int(sk[-1]-sk[0])}/2s')

    def update(_):
        xyz = latest[0]
        if xyz is not None and not view['pause']:
            z = xyz[:, 2] * view['flip']
            sc1.set_offsets(xyz[:, :2]); sc1.set_array(z)
            sc2.set_offsets(np.c_[xyz[:, 0], z]); sc2.set_array(z)
        s = stats() + ('   [paused]' if view['pause'] else '')
        if state['robot']:
            s += '\nrobot side: ' + state['robot']
        if state['err']:
            s += '\n' + state['err']
        txt.set_text(s)

    anim = FuncAnimation(fig, update, interval=33, cache_frame_data=False)  # noqa: F841
    try:
        plt.show()
    finally:
        proc.kill()          # closing ssh breaks the robot-side pipe; the tap exits itself
        log.close()


if __name__ == '__main__':
    sys.path.insert(0, HERE)
    main()
