# AGENTS.md: recreating the H1-2 proximity-skin scene

This repo holds everything needed to rebuild, run and extend a simulated
**Unitree H1-2 humanoid wearing VL53L5CX time-of-flight proximity skins**, and
to plug it into the Correll Lab's GOLEM stack. The skins are HIRO-group's. The
robot model is Unitree's, via the Correll Lab. Nothing third-party is stored
here: `setup_scene.py` fetches it at pinned commits and builds the rest.

Follow the steps in order. Each has a check with the output to expect. If a
check fails, stop and fix it before going on.

## 0. What you are rebuilding

- **The robot:** the H1-2 with Magpie grippers, as a MuJoCo model, the pelvis
  welded in place for the arm scenes.
- **The skins:** HIRO's printed shells on the arms (shoulder cap, upper arm,
  forearm) and knees. Each carries VL53L5CX sensors (8×8 zones, 45° field of
  view, 4 m range, about 12.4 Hz). Sensor *i* here is sensor *i* in HIRO's
  firmware.
- **The sensor model:** noise, invalid-zone flicker and frame rate, fitted from
  real VL53L5CX recordings.
- **The obstacle scene** (`simulate.py`): the right arm loops a reach-and-sweep
  over a table. A person walks up and reaches in beside the forearm. The arm
  slows, holds while the hand is close, then resumes. The person is visual
  only; the skin and a MID-360-like lidar sense them.
- **Training data and a baseline model:** randomised episodes, after ARMOR
  (arXiv 2412.00396), and a small CNN that tells self-returns from environment
  and estimates clearance. It's scored against no-learning baselines, including
  HIRO's sphere self-filter.

## 1. Environment

- **git** and **git-lfs** on PATH (`git lfs version` must work).
- **Python with the packages in `requirements.txt`.** It was tested on CPython
  3.14 with mujoco 3.17 on Windows 11, and on Linux the code has nothing
  platform-specific.
  ```bash
  python -m pip install -r requirements.txt     # torch: CPU build is enough, see the file
  ```
- **On Windows, avoid Anaconda Python 3.10 with mujoco 3.3.1:** its plugin DLLs
  failed to load (`WinError 1114`), and it then crashed on any model. Use a
  recent CPython.
- **A display**, for `--viewer` runs only. Everything else runs headless:
  offscreen rendering goes through `mujoco.Renderer`.

**Check:** `python -c "import mujoco, numpy, yaml; print(mujoco.__version__)"` prints a version.

## 2. Build the scene

```bash
python setup_scene.py            # about 2 min; add --hf to also fit the full-suit sensor model
```

It does the following:
1. **Fetches the robot model.** It clones `correlllab/CL_Assets` at
   `963c93f8611a193eeb344b41b34c44dcf201e0ec` into `./CL_Assets/`, and pulls
   from Git LFS only `meshes/h1_2/*` and `meshes/magpie/*`.
2. **Fetches HIRO's files.** It clones `HIRO-group/proximity_point_motion` as a
   bare, blobless repo, and **extracts** from commit
   `27f2bc8680df5d0f16fce34a736e7a90a5f4d405` (branch `humanoid`): `sensors_config.yaml`, `skins_xacros/*.xacro`,
   `meshes/skin/*.stl` and `raw_data_*.npz`. They go into
   `./third_party/proximity_point_motion/`.
   - There's no checkout, on purpose: the repo has a folder named
     `points:stls`, which Windows can't create.
3. **Imports the skins** (`scripts/import_hiro_skins.py`):
   - re-expresses HIRO's skin STLs, exported in the robot's world frame with
     every joint at zero, in each skin's own frame;
   - copies each sensor pose verbatim from the xacros;
   - writes `config/h12_skin.yaml` and `meshes/skin/*.stl`.
4. **Fits leg spheres** (`scripts/fit_spheres.py`). GOLEM's H1-2 sphere model
   has none on the legs. Writes `config/h12_leg_spheres.yaml`.
5. **Fits the sensor model** (`python -m golem_proximity_skin.sensor_model fit`)
   from HIRO's 10 public recordings, into `config/vl53l5cx_model.yaml`.
   - With `--hf`, it also fits `config/vl53l5cx_model_suit.yaml` from the gated
     Hugging Face dataset `nickljal/proxHumanoid` (`Jul-22`, the three 48–56-sensor
     sessions). That needs an account granted access, and `hf auth login`.
6. **Runs the tests.**

**Check:** the script ends with `41 passed` and `Scene ready.` Re-running is
safe, since finished steps are skipped; `--force` redoes everything.

**Expected import output** (sensors sit within a few mm of their shell):
```
right_knee       right_knee_link            port 5012 R5  5 sensors,  1.0- 4.9 mm from the mesh
right_shoulder   right_shoulder_roll_link   port 5005 R1  6 sensors,  3.8- 4.1 mm from the mesh
right_upper_arm  right_shoulder_yaw_link    port 5007 R2  6 sensors,  3.8- 4.1 mm from the mesh
right_forearm    right_elbow_link           port 5009 R3  8 sensors,  3.8- 4.4 mm from the mesh
(left side mirrored; left_shoulder and left_knee are imported but disabled, see section 6)
```

## 3. Run the scene

All commands run from `core_ws/src/golem_proximity_skin/`.

```bash
python -m golem_proximity_skin.simulate --arm right --viewer          # live, real time
python -m golem_proximity_skin.simulate --arm right                   # skin_sim.mp4 / .png / .npz
python -m golem_proximity_skin.simulate --arm right --no-react        # the arm ignores the skin
python -m golem_proximity_skin.preview --mjcf ../../../CL_Assets/mujoco_assets/scene_h1_2_magpie.xml \
    --layout config/h12_skin.yaml --out skin_preview.png             # still image, all skins
```

**Check:** a headless run (`--video "" --record "" --plot ""`) ends with:
```
right arm: 3 skins, 20 VL53L5CX sensors at 12.4 Hz (fitted sensor model); ...
arm held (speed < 2%) for 2.0 s, first at t = 6.46 s, resumed by t = 8.47 s
```
The timing can move by about 0.01 s with a different sensor-model fit.

**What's simulated:**
- **Physics:** 500 Hz, PD control with gravity compensation on every hinge.
- **Skin:** cast at the sensor model's rate.
- **Lidar:** 10 Hz.
- **Arm speed:** scales from 100% when the nearest non-robot skin return is
  0.25 m or more away, down to 0 at 0.10 m or less.

## 4. Training data and the baseline model

```bash
python -m golem_proximity_skin.generate --out ../../../data/arm_v2 \
    --skins right_shoulder right_upper_arm right_forearm --episodes 1200
python -m golem_proximity_skin.dataset summary ../../../data/arm_v2
python -m golem_proximity_skin.baseline baselines ../../../data/arm_v2        # no-learning filters only
python -m golem_proximity_skin.baseline eval ../../../data/arm_v2 --model ../../../models/arm_v2_baseline
python -m golem_proximity_skin.baseline train ../../../data/arm_v2 --out ../../../models/my_model
```

**Episodes:**
- **Seeded:** episode *i* uses seed *i*.
- **Types:** `avoid`, `dynamic` and `free`, after ARMOR.
- **Randomised:** obstacles, a table and wall, sensor mounting error (±2 mm,
  ±2°), and dead sensors.
- **Masked zones:** each episode applies the board map's masked zones (see
  section 6).

`generate.py`'s docstring lists every array. Inputs are what the hardware gives:
`readings_mm`, `raw_mm`, `q` and `qd`. Labels are what only the simulation
knows: `ideal_mm`, `hit_class`, `clearance_m`, poses and so on. On a 12-core PC
it runs at about 11× real time and 0.3 MB per 8 s episode.

**`models/`** holds two trained baselines with their `test_report.json`:
- **`arm_v1_baseline`** was trained on data generated with an earlier sensor
  model, fitted on fewer recordings.
- **`arm_v2_baseline`** was trained on the single-board model fitted from HIRO's
  public recordings, the Hugging Face single-board sessions, and one board
  recording that isn't in this repo.

The data they were scored on isn't included, so regenerate it as above.
Because the sensor model fitted here differs slightly, a re-evaluation lands
close to, not exactly on, the stored numbers:

| Held-out test (120 episodes, simulation) | arm_v2_baseline | Sphere filter (HIRO's method, GOLEM spheres, 50 mm) |
|---|---|---|
| Self vs. environment, valid zones | 98.2% | 98.9% |
| Clearance per skin, average error | 8.6 cm | – |
| Close calls (<10 cm) caught / alarms correct | 52% / 69% | – |

## 5. Plug into GOLEM (ROS 2, RoboCasa)

`golem_integration/golem_tof_skin.patch` applies to `correlllab/GOLEM` at
`85fe82646ecfae50d365c1f88549407a7ba71e4b`. It touches three files:
- `golem_robocasa/tof_skin_sensors.py` (new): an opt-in ToF skin suite;
- `golem_robocasa/h12_mujoco.py`: a two-line hook;
- `docker/docker-compose.yml`: `GOLEM_TOF_SKIN`, `GOLEM_TOF_SKIN_CONFIG`, and a
  read-only mount of the package into the sim container.

```bash
git clone https://github.com/correlllab/GOLEM.git && cd GOLEM
git checkout 85fe82646ecfae50d365c1f88549407a7ba71e4b
git submodule update --init --recursive && git lfs install
git apply ../h12-proximity-skin/golem_integration/golem_tof_skin.patch
cp -r ../h12-proximity-skin/core_ws/src/golem_proximity_skin core_ws/src/     # after setup_scene.py
```

Then follow GOLEM's `CLAUDE.md`, section 9 (x86 with an NVIDIA GPU, Docker):
```bash
GOLEM_TOF_SKIN=1 docker/scripts/docker_run.sh robocasa      # sim publishes /skin/<name>/tof
docker/scripts/docker_run.sh ros
ros2 launch golem_bringup h12_sim.launch.py use_rviz:=false
ros2 launch golem_proximity_skin skin.launch.py              # skin TF, clouds, monitor, RViz
```

`GOLEM_TOF_SKIN` is off by default, so GOLEM's topic surface and its golden
file don't change. The package's `README.md` lists the topics. This GOLEM path
was **not** run end to end on the authoring machine, which was Windows with no
Docker GPU; only the offline MuJoCo paths above were.

## 6. Known issues and state

- **`left_shoulder` and `left_knee` are disabled** in `config/h12_skin.yaml`,
  because HIRO's own data contradicts itself:
  - `shoulder_roll_L_skin.xacro` reuses the right shoulder's poses unmirrored,
    so sensors 0–3 look into the cap;
  - `knee_L_skin.xacro` gives sensors 0 and 1 an rpy 90° off their listed
    normals.
- **The knee sensors' centre rays hit their own shell mesh** 0.6–10 mm out. On
  the real knee, check whether the shell blocks them. In simulation a sensor's
  own link and skin are excluded from its rays.
- **The forearm skin has no sensor along the forearm's axis**, so a hand
  approaching head-on from where the forearm points is unseen. The scene's
  person reaches in from the side for that reason.
- **The skin has blind gaps between sensors** (45° each): the lidar often sees
  an approach about 0.3 s before the skin does.
- **Self-returns:** about 17% of arm-skin returns are the robot itself (torso,
  other arm, legs). The hardware can't tell them apart; use the sphere filter
  or a learned model.
- **`config/hiro_boards.yaml` describes the author's real shoulder-yaw board**
  (4 of the skin's 6 sensors, firmware 0–3, on HIRO's UDP port 5007):
  - **Its sensor map is marked `verified: false`.** A later check contradicted
    it.
  - **Its masked zones** (12 on firmware 1, where the shell clips the view)
    were measured on that physical board. For another skin, re-map with
    `hiro_udp identify` and re-measure with `hiro_udp mask`.
- **Sensor model provenance:** the fit here uses HIRO's public recordings. The
  author's fit also used gated Hugging Face sessions and one private
  recording. With many sensors running at once, about 60% of zones flag
  invalid, against about 20% on single boards. `--hf` fits that suit model
  separately.

## 7. File map

| Path | What |
|---|---|
| `setup_scene.py` | fetch at pinned commits, build, test |
| `core_ws/src/golem_proximity_skin/` | the ROS 2 / Python package (GOLEM layout); its `README.md` covers every tool |
| `.../golem_proximity_skin/layout.py`, `vl53l5cx.py` | skin layout format; VL53L5CX zone model (HIRO's `proximity_common.py` geometry) |
| `.../mujoco_tof.py`, `sensor_model.py` | ray-cast caster; fitted noise/flicker model |
| `.../simulate.py`, `preview.py` | the obstacle scene; still previews |
| `.../generate.py`, `dataset.py`, `baseline.py`, `sphere_filter.py` | training data, loader, baseline model, sphere self-filter |
| `.../hiro_udp.py`, `hiro_view.py` | real HIRO-firmware boards over UDP: status, identify, record, mask, twin, live view |
| `.../nodes.py`, `launch/`, `rviz/` | ROS 2 nodes for GOLEM |
| `.../scripts/` | importer (HIRO skins), leg-sphere fitter |
| `golem_integration/golem_tof_skin.patch` | the GOLEM-side changes |
| `models/` | trained baselines, with test reports |

## 8. Licences and provenance

- **This repo's code:** written for this project; not licensed for
  redistribution beyond the repo's collaborators until its owner chooses a
  licence.
- **H1-2 model and meshes (CL_Assets):** Unitree, BSD-3. Fetched, not stored.
- **HIRO-group skins, sensor poses and recordings:** their repo has no licence
  file. Fetched, not stored here; ask HIRO-group before redistributing.
- **GOLEM:** the Correll Lab's; no licence file. Only a patch against it is
  stored here.
- **`nickljal/proxHumanoid`:** gated by its owner. Never re-share its files.
