"""Feed Nick's visualizer a thinned lidar cloud, with no Docker and no robot changes.

Stands in for scripts/start_golem_ros.sh + scripts/run_lidar_tof.sh from
proximity_point_motion (humanoid-2). Instead of a golem_ros container joining
the robot's ROS network and pulling every full scan over the Wi-Fi, a
temporary read-only tap on the robot (robot_link.py) thins each scan there and
sends /livox/points and /joint_states down the ssh pipe. This serves them on
127.0.0.1:7420 in sensor_relay.py's format, so main.py runs unchanged:

    terminal 1:  python3 thin_relay.py --radius 4 --pts 3000
    terminal 2:  cd proximity_point_motion   # humanoid-2
                 python3 main.py --lidar --lidar-topic /livox/points \
                     --live-joints --joint-topic /joint_states \
                     --relay 127.0.0.1:7420 --no-camera --no-record

The PC never joins the robot's ROS network, so the robot's nodes do not print
"invalid data size". main.py still needs a local ROS install to decode messages.
Frames are passed on latest-only per topic, so a slow viewer skips stale scans
instead of building a backlog. Ctrl-C stops it; the robot side exits with it.
"""
import argparse, socket, struct, sys, threading, time, os
import robot_link

HDR = struct.Struct('<HIQ')


class Clients:
    def __init__(self):
        self.cv = threading.Condition()
        self.socks = {}                          # sock -> {topic: frame}
        self.skipped = 0

    def add(self, s):
        with self.cv:
            self.socks[s] = {}
        threading.Thread(target=self._send, args=(s,), daemon=True).start()

    def push(self, topic, frame):
        with self.cv:
            for slots in self.socks.values():
                self.skipped += topic in slots
                slots[topic] = frame
            self.cv.notify_all()

    def _send(self, s):
        try:
            while True:
                with self.cv:
                    while s in self.socks and not self.socks[s]:
                        self.cv.wait(0.5)
                    if s not in self.socks:
                        return
                    frames, self.socks[s] = list(self.socks[s].values()), {}
                for f in frames:
                    s.sendall(f)
        except OSError:
            pass
        finally:
            with self.cv:
                self.socks.pop(s, None)
            s.close()
            print('viewer disconnected', flush=True)


def serve(port, clients):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', port)); srv.listen(4)
    while True:
        c, addr = srv.accept()
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        clients.add(c)
        print(f'viewer connected from {addr[0]}:{addr[1]}', flush=True)


def pump(stream, clients, counts):
    """Read sensor_relay frames from the tap and hand each to the viewers."""
    while True:
        h = robot_link.read_exact(stream, HDR.size)
        tlen, plen, _ = HDR.unpack(h)
        topic = robot_link.read_exact(stream, tlen)
        payload = robot_link.read_exact(stream, plen)
        name = topic.decode()
        counts[name] = counts.get(name, 0) + 1
        clients.push(name, h + topic + payload)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, default=7420)
    ap.add_argument('--joint-hz', type=float, default=50.0,
                    help='/joint_states rate built from /lowstate (0 = lidar only)')
    robot_link.add_args(ap, max_pts=3000)
    a = ap.parse_args()

    clients = Clients()
    threading.Thread(target=serve, args=(a.port, clients), daemon=True).start()
    proc = robot_link.start_tap(a, 'relay', joint_hz=a.joint_hz)
    counts = {}

    def errs():
        for line in proc.stderr:
            print('robot', line.decode(errors='replace').rstrip(), flush=True)
    threading.Thread(target=errs, daemon=True).start()

    def report():
        last = {}
        while True:
            time.sleep(5)
            rates = ', '.join(f'{k} {(v - last.get(k, 0)) / 5:.1f} Hz' for k, v in counts.items())
            last = dict(counts)
            print(f'relay: {rates or "no frames yet"}; {len(clients.socks)} viewer(s), '
                  f'{clients.skipped} stale frames skipped', flush=True)
    threading.Thread(target=report, daemon=True).start()
    print(f'serving thinned robot topics on 127.0.0.1:{a.port}', flush=True)
    try:
        pump(proc.stdout, clients, counts)
        print('robot stream ended', flush=True)
    except EOFError:
        print('robot stream ended', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        proc.kill()


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
