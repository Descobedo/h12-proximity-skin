"""Rebuild the H1-2 proximity-skin scene from its sources, at pinned versions.

    python setup_scene.py            # fetch, import, fit, test
    python setup_scene.py --hf       # also fit the full-suit sensor model (gated HF data)

Nothing third-party is stored in this repo. This script fetches it:
  CL_Assets  (correlllab)        the Unitree H1-2 MJCF, URDFs and meshes (Git LFS), BSD-3
  proximity_point_motion (HIRO)  the H1-2 skin meshes, sensor xacros, sensors_config.yaml,
                                 and real VL53L5CX recordings (raw_data_*.npz), branch humanoid
then builds what the package needs from them:
  config/h12_skin.yaml, meshes/skin/*.stl   scripts/import_hiro_skins.py (HIRO's skins, link frames)
  config/h12_leg_spheres.yaml               scripts/fit_spheres.py (leg spheres for the self-filter)
  config/vl53l5cx_model.yaml                sensor_model fit on HIRO's public recordings
  config/vl53l5cx_model_suit.yaml           with --hf: fit on nickljal/proxHumanoid Jul-22 (gated)
and runs the package's tests. Needs git, git-lfs and the Python packages in requirements.txt.
Re-running is safe: finished steps are skipped (use --force to redo them).
"""
import argparse
import glob
import os
import shutil
import stat
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, 'core_ws', 'src', 'golem_proximity_skin')
THIRD = os.path.join(ROOT, 'third_party')

CL_ASSETS = ('https://github.com/correlllab/CL_Assets.git', '963c93f8611a193eeb344b41b34c44dcf201e0ec')
HIRO = ('https://github.com/HIRO-group/proximity_point_motion.git', '27f2bc8680df5d0f16fce34a736e7a90a5f4d405')
HF_DATASET = 'nickljal/proxHumanoid'
# The Jul-22 sessions with 48-56 sensors streaming at once: the full suit.
HF_SUIT = ('raw_data_20260722_095804.npz', 'raw_data_20260722_095949.npz', 'raw_data_20260722_100156.npz')


def rmtree(path):
    """shutil.rmtree that also removes read-only files (git's objects on Windows)."""
    def clear(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onexc=clear) if sys.version_info >= (3, 12) else shutil.rmtree(path, onerror=clear)


def run(cmd, cwd=None, env=None):
    print('$', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def fetch_cl_assets(force):
    dst = os.path.join(ROOT, 'CL_Assets')         # where the package's default paths look
    marker = os.path.join(dst, 'mujoco_assets', 'scene_h1_2_magpie_pelvis_fixed.xml')
    if os.path.exists(marker) and not force:
        print('CL_Assets: present')
        return
    if os.path.exists(dst):
        rmtree(dst)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    run(['git', 'clone', '--quiet', CL_ASSETS[0], dst], env=env)
    run(['git', 'checkout', '--quiet', CL_ASSETS[1]], cwd=dst, env=env)
    run(['git', 'lfs', 'pull', '--include', 'meshes/h1_2/*,meshes/magpie/*'], cwd=dst)
    head = open(glob.glob(os.path.join(dst, 'meshes', 'h1_2', '*.STL'))[0], 'rb').read(40)
    if head.startswith(b'version https://git-lfs'):
        sys.exit('CL_Assets meshes are still LFS pointers: is git-lfs installed?')


def fetch_hiro(force):
    """The files the scene needs from HIRO's repo at the pinned commit, extracted from git's
    object store without a checkout: the repo holds a 'points:stls' folder, a name Windows
    cannot create, and git refuses to check out a tree that contains one."""
    dst = os.path.join(THIRD, 'proximity_point_motion')
    if os.path.exists(os.path.join(dst, 'sensors_config.yaml')) and not force:
        print('proximity_point_motion: present')
        return dst
    if os.path.exists(dst):
        rmtree(dst)
    os.makedirs(THIRD, exist_ok=True)
    git = os.path.join(THIRD, 'proximity_point_motion.git')
    if not os.path.exists(git):
        run(['git', 'clone', '--quiet', '--bare', '--filter=blob:none', HIRO[0], git])
    names = subprocess.run(['git', 'ls-tree', '-r', '--name-only', HIRO[1]], cwd=git, check=True,
                           capture_output=True, text=True).stdout.splitlines()
    wanted = [n for n in names if n == 'sensors_config.yaml'
              or (n.startswith('raw_data_') and n.endswith('.npz') and '/' not in n)
              or (n.startswith('skins_xacros/') and n.endswith('.xacro'))
              or (n.startswith('meshes/skin/') and n.endswith('.stl') and n.count('/') == 2)]
    print(f'extracting {len(wanted)} files from proximity_point_motion@{HIRO[1][:8]}', flush=True)
    for n in wanted:
        blob = subprocess.run(['git', 'cat-file', 'blob', f'{HIRO[1]}:{n}'], cwd=git, check=True,
                              capture_output=True).stdout
        out = os.path.join(dst, *n.split('/'))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'wb') as f:
            f.write(blob)
    return dst


def build(hiro, force, hf):
    py = sys.executable
    mjcf = os.path.join(ROOT, 'CL_Assets', 'mujoco_assets', 'h1_2_magpie.xml')
    scene = os.path.join(ROOT, 'CL_Assets', 'mujoco_assets', 'scene_h1_2_magpie_pelvis_fixed.xml')
    cfg = os.path.join(PKG, 'config')
    if force or not os.path.exists(os.path.join(cfg, 'h12_skin.yaml')):
        run([py, os.path.join('scripts', 'import_hiro_skins.py'), hiro, mjcf], cwd=PKG)
    if force or not os.path.exists(os.path.join(cfg, 'h12_leg_spheres.yaml')):
        run([py, os.path.join('scripts', 'fit_spheres.py'), scene, '--out',
             os.path.join('config', 'h12_leg_spheres.yaml'), '--max-spheres', '24'], cwd=PKG)
    if force or not os.path.exists(os.path.join(cfg, 'vl53l5cx_model.yaml')):
        recs = sorted(glob.glob(os.path.join(hiro, 'raw_data_*.npz')))
        run([py, '-m', 'golem_proximity_skin.sensor_model', 'fit', *recs,
             '--out', os.path.join('config', 'vl53l5cx_model.yaml')], cwd=PKG)
    if hf and (force or not os.path.exists(os.path.join(cfg, 'vl53l5cx_model_suit.yaml'))):
        from huggingface_hub import hf_hub_download   # needs access to the gated dataset
        files = [hf_hub_download(HF_DATASET, f'Jul-22/{f}', repo_type='dataset',
                                 local_dir=os.path.join(THIRD, 'hf_proxHumanoid')) for f in HF_SUIT]
        run([py, '-m', 'golem_proximity_skin.sensor_model', 'fit', *files,
             '--out', os.path.join('config', 'vl53l5cx_model_suit.yaml')], cwd=PKG)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--hf', action='store_true', help='also fit the full-suit model from the gated HF data')
    ap.add_argument('--force', action='store_true', help='redo every step')
    ap.add_argument('--skip-tests', action='store_true')
    args = ap.parse_args()
    for tool in ('git', 'git-lfs'):
        if shutil.which(tool) is None:
            sys.exit(f'{tool} not found on PATH')
    fetch_cl_assets(args.force)
    hiro = fetch_hiro(args.force)
    build(hiro, args.force, args.hf)
    if not args.skip_tests:
        run([sys.executable, '-m', 'pytest', 'test', '-q', '-p', 'no:cacheprovider'], cwd=PKG)
    print('\nScene ready. Try:\n'
          f'  cd {os.path.relpath(PKG)}\n'
          '  python -m golem_proximity_skin.simulate --arm right --viewer')


if __name__ == '__main__':
    main()
