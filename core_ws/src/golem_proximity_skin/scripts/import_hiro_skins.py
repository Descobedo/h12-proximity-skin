"""Import HIRO-group/proximity_point_motion's H1-2 skins into this package.

    python3 scripts/import_hiro_skins.py <proximity_point_motion checkout, branch humanoid> \
        ../../../CL_Assets/mujoco_assets/h1_2_magpie.xml [skin_name ...]

Reads HIRO's sensors_config.yaml (which skins exist, their H1-2 link, UDP port
and label) and each skin's xacro in skins_xacros/ (the dermis origin in its
link, and each sensor's origin in the dermis frame, +z = boresight). Both are
copied as HIRO has them, so sensor <i> here is sensor <i> in HIRO's firmware
and tools.

HIRO's skin STLs are exported in the robot's world frame with every joint at
zero and the pelvis at the origin (sensors_config.yaml: mesh_in_world_frame).
Here each mesh is re-expressed in its dermis frame, `<name>_skin`, so it rides
its link like the sensors do. That is world = T_link @ T_dermis @ p, as the
knee xacros spell out.

Default skins: the arms and knees. Writes meshes/skin/<name>.stl and
config/h12_skin.yaml; prints how far each sensor sits from its mesh's surface.
Needs numpy, pyyaml and mujoco (for the zero-pose kinematics).
"""
import os
import struct
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
from golem_proximity_skin.layout import rpy_to_matrix  # noqa: E402

# HIRO skin name -> name here.
NAMES = {
    'shoulder_roll_R_skin': 'right_shoulder', 'shoulder_yaw_R_skin': 'right_upper_arm',
    'elbow_R_skin': 'right_forearm', 'knee_R_skin': 'right_knee',
    'shoulder_roll_L_skin': 'left_shoulder', 'shoulder_yaw_L_skin': 'left_upper_arm',
    'elbow_L_skin': 'left_forearm', 'knee_L_skin': 'left_knee',
}
# Skins imported but disabled, with why: HIRO's poses for them disagree with their
# own data (checked by casting each sensor's centre ray against its skin mesh).
DISABLED = {
    'left_shoulder': "HIRO's shoulder_roll_L_skin.xacro reuses the right shoulder's sensor "
                     'poses unmirrored: sensors 0-3 look into the cap',
    'left_knee': "HIRO's knee_L_skin.xacro: sensors 0 and 1 have an rpy 90 deg off the "
                 'normal it lists for them',
}
ORDER = ['right_knee', 'right_shoulder', 'right_upper_arm', 'right_forearm',
         'left_knee', 'left_shoulder', 'left_upper_arm', 'left_forearm']


def read_stl(path):
    b = open(path, 'rb').read()
    n = struct.unpack('<I', b[80:84])[0]
    if 84 + 50 * n != len(b):
        raise ValueError(f'{path}: not a binary STL')
    rec = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    return np.frombuffer(b, dtype=rec, count=n, offset=84)['v'].astype(np.float64)


def write_stl(path, tris, header):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-15
    rec = np.zeros(len(tris), dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    rec['n'], rec['v'] = n, tris
    with open(path, 'wb') as f:
        f.write(header.encode()[:80].ljust(80, b' '))
        f.write(struct.pack('<I', len(tris)))
        f.write(rec.tobytes())


def _origin(el):
    o = el.find('origin')
    xyz = [float(v) for v in o.get('xyz', '0 0 0').split()]
    rpy = [float(v) for v in o.get('rpy', '0 0 0').split()]
    return xyz, rpy


def parse_xacro(path):
    """(dermis xyz, rpy), [(sensor number, xyz, rpy, normal)] from one HIRO skin xacro."""
    root = ET.parse(path).getroot()
    tag = lambda e: e.tag.split('}')[-1]  # noqa: E731
    dermis, sensors = None, []
    for el in root.iter():
        if tag(el) == 'dermis_base_macro':
            dermis = _origin(el)
        elif tag(el) == 'self_cap_sensor':
            xyz, rpy = _origin(el)
            normal = [float(el.get(f'normal_{a}')) for a in 'xyz']
            sensors.append((int(el.get('sensor_number')), xyz, rpy, normal))
    if dermis is None or not sensors:
        raise ValueError(f'{path}: no dermis or sensors')
    sensors.sort(key=lambda s: s[0])
    if [s[0] for s in sensors] != list(range(len(sensors))):
        raise ValueError(f'{path}: sensor numbers are not 0..n-1')
    return dermis, sensors


def link_in_world(model, data, link):
    """Pose of `link` in the pelvis frame at qpos0 (HIRO's mesh world frame)."""
    p = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
    Rp = data.xmat[p].reshape(3, 3)
    return Rp.T @ (data.xpos[b] - data.xpos[p]), Rp.T @ data.xmat[b].reshape(3, 3)


def fmt(v):
    return '[' + ', '.join(f'{x:.4f}' for x in v) + ']'


def main(src, mjcf, wanted):
    cfg = yaml.safe_load(open(os.path.join(src, 'sensors_config.yaml'), encoding='utf-8'))
    entries = {NAMES[s['skin_name']]: s for s in cfg['skins'] if s['skin_name'] in NAMES}
    wanted = wanted or [n for n in ORDER if n in entries]
    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)     # qpos0: every joint at zero
    os.makedirs(os.path.join(PKG, 'meshes', 'skin'), exist_ok=True)
    for old in os.listdir(os.path.join(PKG, 'meshes', 'skin')):
        if old.endswith('.stl'):
            os.remove(os.path.join(PKG, 'meshes', 'skin', old))
    head = [
        "# Proximity skins on the H1-2: HIRO-group/proximity_point_motion's skins",
        '# (branch `humanoid`: sensors_config.yaml, skins_xacros/, meshes/skin/), imported',
        '# by scripts/import_hiro_skins.py. Do not edit by hand: re-run the script.',
        "# Each skin's sensors are HIRO's, numbered as HIRO's firmware sends them;",
        '# `port` / `label` are its UDP port and label prefix there. Origins: the',
        "# dermis frame in its link; sensors in the dermis frame (+z = boresight).",
        '# Set `enabled: false` on a skin to drop it.',
        '#',
        '# Read by the ROS nodes, the HIRO UDP bridge (golem_proximity_skin.hiro_udp), the',
        "# offline simulation and the RoboCasa sim's ToF model (GOLEM_TOF_SKIN=1).",
        '',
        'rate_hz: 12.5        # HIRO firmware, measured (12.3-12.5 Hz per sensor); VL53L5CX max is 15',
        '',
        'skins:',
    ]
    lines = list(head)
    for name in wanted:
        e = entries[name]
        (dxyz, drpy), sensors = parse_xacro(os.path.join(src, e['xacro_file']))
        link = e['h1_link']
        pl, Rl = link_in_world(model, data, link)
        Rd = rpy_to_matrix(drpy)
        tris = (read_stl(os.path.join(src, e['stl_path'])) - pl) @ Rl     # world -> link
        tris = (tris - np.array(dxyz)) @ Rd                               # link -> dermis
        out = f'{name}.stl'
        write_stl(os.path.join(PKG, 'meshes', 'skin', out), tris,
                  f'golem_proximity_skin: HIRO {os.path.basename(e["stl_path"])} in {name}_skin')
        cent = tris.mean(axis=1)
        off, align = [], []
        nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-15
        for _, xyz, rpy, normal in sensors:
            d = np.linalg.norm(cent - np.array(xyz), axis=1)
            off.append(d.min())
            n = nrm[np.argsort(d)[:40]].mean(axis=0)
            align.append(abs(rpy_to_matrix(rpy)[:, 2] @ (n / np.linalg.norm(n))))
        print(f'{name:16s} {link:26s} port {e["port"]} {e["label_prefix"]:3s} {len(sensors)} sensors, '
              f'{min(off) * 1000:4.1f}-{max(off) * 1000:4.1f} mm from the mesh, '
              f'|boresight . normal| >= {min(align):.2f}')
        lines += [
            f'  - name: {name}',
            f'    parent: {link}',
            f'    port: {int(e["port"])}',
            f'    label: {e["label_prefix"]}',
        ]
        if name in DISABLED:
            lines += [f'    # Disabled: {DISABLED[name]}.', '    enabled: false']
        lines += [
            f'    origin: {{xyz: {fmt(dxyz)}, rpy: {fmt(drpy)}}}',
            f'    mesh: package://golem_proximity_skin/meshes/skin/{out}',
            '    sensors:',
        ]
        lines += [f'      - {{xyz: {fmt(xyz)}, rpy: {fmt(rpy)}}}' for _, xyz, rpy, _ in sensors]
    with open(os.path.join(PKG, 'config', 'h12_skin.yaml'), 'w', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote config/h12_skin.yaml and meshes/skin/*.stl')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
