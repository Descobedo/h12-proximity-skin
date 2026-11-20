# Runs ON THE ROBOT for as long as the PC keeps the ssh pipe open. It is sent
# over ssh stdin and never written to disk there; nothing on the robot is
# installed, configured or restarted. It is a read-only listener, like
# `ros2 topic echo`: best-effort QoS so it can never slow the drivers, no
# rosout, no parameter services, and it publishes nothing on the robot's ROS
# network. It thins each lidar frame ON the robot, so only the thinned cloud
# crosses the Wi-Fi, and writes it to stdout. It exits when the pipe closes,
# and `timeout` on the launching command caps how long it can run.
#
# Settings (environment, set by the PC-side launcher):
#   TAP_FMT      view  -> compact frames for lidar_view.py (Windows, no ROS)
#                relay -> sensor_relay.py frames for Nick's main.py --relay
#   TAP_MAX_PTS  points sent per frame (default 4000)
#   TAP_RADIUS   drop points farther than this from the lidar, m (0 = keep all)
#   TAP_NEAR     points within this range are kept first, m (default 1.5)
#   TAP_HZ       lidar frames sent per second, at most (0 = every frame)
#   TAP_JOINT_HZ relay only: /lowstate -> /joint_states at this rate (0 = off)
import os, struct, sys, threading, time
import numpy as np

PT = np.dtype([('t', '<u4'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
               ('r', 'u1'), ('tag', 'u1'), ('line', 'u1'), ('pad', 'u1')])
VIEW_HDR = struct.Struct('<4sIdddIII')    # magic seq stamp recv sent n_total n_sent skipped
RELAY_HDR = struct.Struct('<HIQ')         # sensor_relay.py: topic len, payload len, recv ns

H1_2_JOINT_NAMES = [
    'left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_knee_joint',
    'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_yaw_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_knee_joint',
    'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'torso_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint',
    'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint',
    'right_wrist_pitch_joint', 'right_wrist_yaw_joint']


def settings(env=os.environ):
    return {'fmt': env.get('TAP_FMT', 'view'),
            'max_pts': int(env.get('TAP_MAX_PTS', '4000')),
            'radius': float(env.get('TAP_RADIUS', '0')),
            'near': float(env.get('TAP_NEAR', '1.5')),
            'hz': float(env.get('TAP_HZ', '0')),
            'joint_hz': float(env.get('TAP_JOINT_HZ', '50'))}


def parse_custom(buf):
    """(frame_id, sec, nsec, points) from one livox CustomMsg CDR buffer.

    The points are the trailing array of 20-byte records; its uint32 length
    sits right before it. Checking that length against the buffer size finds
    the start without relying on the frame_id length.
    """
    sec, nsec = struct.unpack_from('<iI', buf, 4)
    flen = struct.unpack_from('<I', buf, 12)[0]
    frame = bytes(buf[16:16 + max(flen - 1, 0)]).decode('utf-8', 'replace')
    for off in (48, 44, 52, 56, 40, 60, 64):
        if off + 4 > len(buf):
            continue
        n = struct.unpack_from('<I', buf, off - 4)[0]
        if off + n * 20 == len(buf):
            return frame, sec, nsec, np.frombuffer(buf, PT, n, off)
    return None


def thin(pts, cfg, rng):
    """xyz float32 (N,3): drop empty returns, crop, then keep near points first."""
    xyz = np.stack([pts['x'], pts['y'], pts['z']], 1).astype(np.float32)
    d2 = (xyz * xyz).sum(1)
    keep = d2 > 1e-6
    if cfg['radius'] > 0:
        keep &= d2 <= cfg['radius'] ** 2
    xyz, d2 = xyz[keep], d2[keep]
    m = cfg['max_pts']
    if m <= 0 or len(xyz) <= m:
        return xyz
    near = np.flatnonzero(d2 <= cfg['near'] ** 2)
    if len(near) >= m:                       # crowded close up: keep the closest m
        return xyz[near[np.argsort(d2[near])[:m]]]
    far = np.flatnonzero(d2 > cfg['near'] ** 2)
    far = rng.choice(far, m - len(near), replace=False)
    return xyz[np.concatenate([near, far])]


def view_frame(seq, stamp, recv, n_total, xyz, skipped):
    return VIEW_HDR.pack(b'LDR1', seq, stamp, recv, time.time(), n_total, len(xyz),
                         skipped) + xyz.astype('<f2').tobytes()


def relay_frame(topic, payload, recv_ns):
    t = topic.encode()
    return RELAY_HDR.pack(len(t), len(payload), recv_ns) + t + payload


class Latest:
    """One slot per stream: a slow link skips stale frames instead of queueing them."""

    def __init__(self):
        self.cv = threading.Condition()
        self.slots = {}
        self.skipped = 0

    def put(self, key, item):
        with self.cv:
            if key in self.slots:
                self.skipped += 1
            self.slots[key] = item
            self.cv.notify()

    def take(self):
        with self.cv:
            while not self.slots:
                self.cv.wait()
            items, self.slots = list(self.slots.values()), {}
            return items


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from livox_ros_driver2.msg import CustomMsg

    cfg = settings()
    out = sys.stdout.buffer
    latest = Latest()
    rng = np.random.default_rng(0)
    st = {'seq': 0, 'last_sent': 0.0, 'pts_in': 0, 'pts_out': 0, 'joints': 0}
    relay = cfg['fmt'] == 'relay'
    if relay:
        from rclpy.serialization import serialize_message, deserialize_message
        from builtin_interfaces.msg import Time
        from sensor_msgs.msg import PointCloud2, PointField, JointState
        from std_msgs.msg import Header
        fields = [PointField(name=c, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, c in enumerate('xyz')]

    def on_lidar(buf):
        recv = time.time()
        if cfg['hz'] > 0 and recv - st['last_sent'] < 1.0 / cfg['hz'] - 0.004:
            return
        p = parse_custom(buf)
        if p is None:
            return
        frame, sec, nsec, pts = p
        st['last_sent'] = recv
        st['seq'] += 1
        xyz = thin(pts, cfg, rng)
        st['pts_in'] += len(pts); st['pts_out'] += len(xyz)
        if relay:
            c = PointCloud2(header=Header(stamp=Time(sec=sec, nanosec=nsec), frame_id=frame),
                            height=1, width=len(xyz), fields=fields, is_bigendian=False,
                            point_step=12, row_step=12 * len(xyz), is_dense=True)
            c.data = xyz.astype('<f4').tobytes()
            latest.put('lidar', relay_frame('/livox/points', serialize_message(c),
                                            int(recv * 1e9)))
        else:
            latest.put('lidar', view_frame(st['seq'], sec + nsec * 1e-9, recv, len(pts),
                                           xyz, latest.skipped))

    def on_lowstate(buf, LowState):
        now = time.time()
        if now - st.get('last_joint', 0.0) < 1.0 / cfg['joint_hz']:
            return                           # decode only the ones we send
        st['last_joint'] = now
        msg = deserialize_message(buf, LowState)
        n = len(H1_2_JOINT_NAMES)
        if len(msg.motor_state) < n:
            return
        sec = int(now); js = JointState()
        js.header.stamp = Time(sec=sec, nanosec=int((now - sec) * 1e9))
        js.name = H1_2_JOINT_NAMES
        js.position = [float(msg.motor_state[i].q) for i in range(n)]
        js.velocity = [float(msg.motor_state[i].dq) for i in range(n)]
        js.effort = [float(msg.motor_state[i].tau_est) for i in range(n)]
        st['joints'] += 1
        latest.put('joints', relay_frame('/joint_states', serialize_message(js), int(now * 1e9)))

    def writer():
        try:
            while True:
                for item in latest.take():
                    out.write(item)
                out.flush()
        except (BrokenPipeError, OSError):
            os._exit(0)                       # PC closed the pipe: leave at once

    def report():
        while True:
            time.sleep(5)
            sys.stderr.write(
                f"tap: {st['seq']} lidar frames, kept {st['pts_out']}/{st['pts_in']} points, "
                f"{st['joints']} joint msgs, {latest.skipped} skipped for the link\n")
            sys.stderr.flush()

    rclpy.init()
    node = Node('thin_tap_readonly', enable_rosout=False, start_parameter_services=False)
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
    node.create_subscription(CustomMsg, '/livox/lidar', on_lidar, qos, raw=True)
    if relay and cfg['joint_hz'] > 0:
        try:
            from unitree_hg.msg import LowState
            node.create_subscription(LowState, '/lowstate',
                                     lambda b: on_lowstate(b, LowState), qos, raw=True)
        except ImportError:
            sys.stderr.write('tap: unitree_hg not found, sending lidar only\n')
    threading.Thread(target=writer, daemon=True).start()
    threading.Thread(target=report, daemon=True).start()
    sys.stderr.write(f'tap: listening, settings {cfg}\n'); sys.stderr.flush()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
