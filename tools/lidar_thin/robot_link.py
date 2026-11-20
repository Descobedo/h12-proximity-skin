"""Start lidar_tap.py on the robot over ssh, as a temporary read-only process.

The tap is sent on ssh stdin, so no file is written on the robot, and it runs
under `timeout`, so it cannot outlive --minutes even if the PC side dies.
Nothing on the robot is installed, configured or restarted.

Login: plink (PuTTY) with the pinned host key if it is on PATH, otherwise
OpenSSH. The password comes from H12_PW (or a prompt) for plink, and from
SSHPASS via sshpass for OpenSSH; with ssh keys set up, none is needed.
"""
import argparse, getpass, os, shutil, subprocess

# Lab defaults; override with H12_HOST / H12_HOSTKEY for another robot.
HOST = os.environ.get('H12_HOST', 'unitree@192.168.123.164')
HOSTKEY = os.environ.get('H12_HOSTKEY', 'SHA256:9sAfqxQxTLI19yIsVX6/3DYqkRryyrxLAnO4oMJepyo')
HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE = ('export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp '
          'CYCLONEDDS_URI=file:///home/unitree/cyclonedds.xml ROS_DOMAIN_ID=0; '
          'source /opt/ros/humble/setup.bash; source ~/GOLEM/core_ws/install/setup.bash; '
          '{env} exec timeout {secs} python3 -u -')


def add_args(ap, max_pts=4000):
    g = ap.add_argument_group('thinning on the robot (nothing is changed there)')
    g.add_argument('--pts', type=int, default=max_pts, help='points per frame sent over Wi-Fi')
    g.add_argument('--radius', type=float, default=0.0, help='drop points beyond this, m (0 = off)')
    g.add_argument('--near', type=float, default=1.5, help='points this close are kept first, m')
    g.add_argument('--hz', type=float, default=0.0, help='lidar frames per second (0 = all 30)')
    g.add_argument('--minutes', type=float, default=30.0, help='hard cap on the robot side')
    g.add_argument('--host', default=HOST)


def start_tap(a, fmt, joint_hz=0.0):
    env = {'TAP_FMT': fmt, 'TAP_MAX_PTS': a.pts, 'TAP_RADIUS': a.radius,
           'TAP_NEAR': a.near, 'TAP_HZ': a.hz, 'TAP_JOINT_HZ': joint_hz}
    remote = REMOTE.format(env=' '.join(f'{k}={v}' for k, v in env.items()),
                           secs=max(60, int(a.minutes * 60)))
    if shutil.which('plink'):
        pw = os.environ.get('H12_PW') or getpass.getpass('robot password: ')
        cmd = ['plink', '-batch', '-ssh', '-T', '-hostkey', HOSTKEY, '-pw', pw, a.host, remote]
    else:
        cmd = ['ssh', '-T', '-o', 'ServerAliveInterval=5', a.host, remote]
        if os.environ.get('SSHPASS') and shutil.which('sshpass'):
            cmd = ['sshpass', '-e'] + cmd
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0)
    with open(os.path.join(HERE, 'lidar_tap.py'), 'rb') as f:
        proc.stdin.write(f.read())
    proc.stdin.close()
    return proc


def read_exact(f, n):
    b = b''
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            raise EOFError
        b += c
    return b
