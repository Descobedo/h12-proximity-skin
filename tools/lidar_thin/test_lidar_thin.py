"""Offline checks (no robot, no ROS): python -m pytest tools/lidar_thin -q"""
import io, os, socket, struct, sys, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidar_tap, thin_relay, robot_link


def custom_msg(xyz, sec=100, nsec=5, frame='livox_frame'):
    """A livox CustomMsg as XCDR1 bytes, laid out as the driver sends it."""
    f = frame.encode() + b'\0'
    b = bytearray(b'\x00\x01\x00\x00') + struct.pack('<iII', sec, nsec, len(f)) + f
    while (len(b) - 4) % 8:
        b += b'\0'
    b += struct.pack('<QIB3x', 123, len(xyz), 0)
    while (len(b) - 4) % 4:
        b += b'\0'
    b += struct.pack('<I', len(xyz))
    pts = np.zeros(len(xyz), lidar_tap.PT)
    pts['x'], pts['y'], pts['z'] = xyz.T
    return bytes(b) + pts.tobytes()


def cfg(**kw):
    c = lidar_tap.settings({}); c.update(kw); return c


def test_parse_matches_the_driver_layout():
    xyz = np.random.default_rng(1).normal(size=(500, 3)).astype(np.float32)
    buf = custom_msg(xyz)
    assert struct.unpack_from('<I', buf, 44)[0] == 500      # same offsets Nick's bridge uses
    frame, sec, nsec, pts = lidar_tap.parse_custom(buf)
    assert (frame, sec, nsec) == ('livox_frame', 100, 5)
    np.testing.assert_array_equal(np.stack([pts['x'], pts['y'], pts['z']], 1), xyz)
    assert lidar_tap.parse_custom(buf[:-3]) is None           # torn frame is refused


def test_thin_keeps_near_points_and_crops():
    rng = np.random.default_rng(2)
    near = rng.uniform(-0.5, 0.5, (300, 3))
    far = rng.uniform(5, 9, (6000, 3))
    xyz = np.vstack([near, far, np.zeros((50, 3))]).astype(np.float32)
    pts = lidar_tap.parse_custom(custom_msg(xyz))[3]
    out = lidar_tap.thin(pts, cfg(max_pts=1000, near=1.5), rng)
    assert len(out) == 1000
    d = np.linalg.norm(out, axis=1)
    assert (d > 1e-3).all()                                   # empty returns dropped
    assert (d <= 1.5).sum() == 300                            # every near point kept
    out = lidar_tap.thin(pts, cfg(max_pts=0, radius=3.0), rng)
    assert len(out) == 300                                    # crop alone
    out = lidar_tap.thin(pts, cfg(max_pts=100, near=1.5), rng)
    assert len(out) == 100 and np.linalg.norm(out, axis=1).max() <= 1.5   # closest first


def test_view_frame_round_trip():
    xyz = np.array([[1, 2, 3], [-1, 0.5, -2]], np.float32)
    f = lidar_tap.view_frame(7, 1.5, 1.6, 6000, xyz, 2)
    h = lidar_tap.VIEW_HDR.unpack(f[:lidar_tap.VIEW_HDR.size])
    assert h[0] == b'LDR1' and h[1] == 7 and h[5:] == (6000, 2, 2)
    back = np.frombuffer(f[lidar_tap.VIEW_HDR.size:], '<f2').reshape(-1, 3)
    np.testing.assert_allclose(back, xyz, atol=1e-2)


def test_latest_skips_stale_frames():
    L = lidar_tap.Latest()
    for i in range(5):
        L.put('lidar', i)
    L.put('joints', 'j')
    assert sorted(map(str, L.take())) == ['4', 'j'] and L.skipped == 4


def test_relay_serves_frames_in_sensor_relay_format():
    stream = io.BytesIO(b''.join([
        lidar_tap.relay_frame('/livox/points', b'P' * 1000, 11),
        lidar_tap.relay_frame('/joint_states', b'J' * 50, 12)]))
    clients = thin_relay.Clients()
    srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1)
    port = srv.getsockname()[1]
    viewer = socket.create_connection(('127.0.0.1', port))
    conn, _ = srv.accept(); clients.add(conn)
    counts = {}
    try:
        thin_relay.pump(stream, clients, counts)
    except EOFError:
        pass
    assert counts == {'/livox/points': 1, '/joint_states': 1}
    viewer.settimeout(3)
    rf = viewer.makefile('rb')
    got = {}
    for _ in range(2):                    # what Nick's SensorRelayClient reads
        tl, pl, ts = struct.unpack('<HIQ', robot_link.read_exact(rf, 14))
        topic = robot_link.read_exact(rf, tl).decode()
        got[topic] = (len(robot_link.read_exact(rf, pl)), ts)
    assert got == {'/livox/points': (1000, 11), '/joint_states': (50, 12)}
    viewer.close(); srv.close()


def test_remote_command_is_read_only_and_capped():
    r = robot_link.REMOTE.format(env='TAP_FMT=view', secs=1800)
    assert 'exec timeout 1800 python3 -u -' in r
    for word in ('daemon', 'kill', 'rm ', 'apt', 'pip', 'sudo', 'systemctl', 'param set', '>'):
        assert word not in r
