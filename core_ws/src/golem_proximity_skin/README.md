# golem_proximity_skin

Proximity skins for the H1-2's arms: VL53L5CX multizone ToF sensors in
printed skin shells, modelled on [HIRO-group/proximity_point_motion](https://github.com/HIRO-group/proximity_point_motion)'s
skins for the FR3. This package has the layout, the skin's frames and RViz
visuals, ToF → point clouds, and a monitor that subscribes to both the skins and
the lidar. The RoboCasa sim reuses the package's ToF model to publish simulated
readings.

## The skin model

Same as proximity_point_motion's `skins.xacro` / `proximity_common.py`:

- A skin is a dermis frame `<name>_skin`, fixed to a robot link, carrying
  sensors `<name>_sensor_<i>` at fixed poses. Each sensor looks along its +z.
- A sensor is an 8×8 grid of zones over a 45° field of view, out to 4 m. Zone
  `k = row * 8 + col`, and the ray directions are proximity_common's, so
  `test_vl53l5cx.py` checks them against that code.

`config/h12_skin.yaml` holds HIRO's H1-2 arm skins, from the `humanoid` branch's
[`meshes/skin`](https://github.com/HIRO-group/proximity_point_motion/tree/humanoid/meshes/skin).
Each arm has three skins:

| Skin | Link | Sensors |
|---|---|---|
| `<side>_shoulder` | `<side>_shoulder_roll_link` | 6 |
| `<side>_upper_arm` | `<side>_shoulder_yaw_link` | 6 |
| `<side>_forearm` | `<side>_elbow_link` | 8 |

That's 20 sensors per arm. `scripts/import_hiro_skins.py` generates the file
and the meshes in `meshes/skin/`; re-run it when HIRO's skins change, rather
than editing them by hand. Its docstring explains the conversion:

- HIRO's STLs are exported in the torso frame with the robot in its zero
  pose. The script re-expresses them in each link's frame.
- The sensor poses are transcribed from HIRO's tables (`points:stls/*.png`),
  which cover the right arm. The left arm is the mirror image.
- `test_layout.py` checks that every sensor sits within 6 mm of its skin's
  surface and looks outward along the surface normal.

For a one-arm setup, set `enabled: false` on the other arm's skins. `layout.py`
documents the format. A skin is an explicit list of sensor poses with an
optional mesh, the same data as a `skins.xacro` macro. A parametric `sleeve`
form remains for quick experiments.

## Topics

| Topic | Type | From |
|---|---|---|
| `/skin/<name>/tof` | `std_msgs/UInt16MultiArray`: dim `sensor` × `zone`, mm, 0 = no return | the skin's hardware bridge; in sim, `golem_robocasa/tof_skin_sensors.py` |
| `/tf_static` | `<parent>` → `<name>_skin` → `<name>_sensor_<i>` | `skin_description` |
| `/skin/description_markers` | `MarkerArray`, latched: skin meshes + sensor boards | `skin_description` |
| `/skin/<name>/points` | `PointCloud2` in `<name>_skin`: `x y z range sensor zone` | `skin_cloud` |
| `/skin/proximity_markers` | `MarkerArray`: each sensor's FOV cut at its nearest return, red (≤ `near_m`) → green (≥ `far_m`) | `skin_monitor` |
| `/skin/nearest_range` | `std_msgs/Float32`, m: the nearest return of any sensor | `skin_monitor` |

`skin_monitor` subscribes to every `/skin/<name>/tof` and to the lidar's
`/livox/pointcloud`, and logs the status of both every 2 s. Code that needs
both streams can import `golem_proximity_skin.listener.SensorListener`. It
keeps the latest of each and calls your callbacks.

## Running it in the sim (x86)

```bash
# Terminal 1: the sim, with the skins on
GOLEM_TOF_SKIN=1 docker/scripts/docker_run.sh robocasa
#   look for: [tof_skin] 6 skins, 40 VL53L5CX sensors at 15 Hz from ...

# Terminal 2: the robot
docker/scripts/docker_run.sh ros
ros2 launch golem_bringup h12_sim.launch.py use_rviz:=false

# Terminal 3: the skins, and an RViz of robot + lidar + skins
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
ros2 launch golem_proximity_skin skin.launch.py
ros2 topic hz /skin/left_forearm/tof        # ~15 Hz
```

`GOLEM_TOF_SKIN` is off by default, so the sim's topic surface (and
`tests/robocasa/surface/h12.json`) does not change. `GOLEM_TOF_SKIN_CONFIG`
points the sim at another layout. Pass the same file to the launch's `layout:=`.
The Mac port does not mount this package and has no skins.

## Simulating one arm at work, without ROS

`simulate.py` runs the whole sensing loop for one skinned arm in MuJoCo. It
needs only `mujoco`, `numpy`, `pyyaml`, `pillow` and, for the outputs,
`matplotlib` and `imageio[ffmpeg]`:

```bash
cd core_ws/src/golem_proximity_skin
python3 -m golem_proximity_skin.simulate --arm right      # -> skin_sim.mp4 / .png / .npz
python3 -m golem_proximity_skin.simulate --arm left --viewer
```

The H1-2 stands with its pelvis welded at a table. PD control with gravity
compensation drives every joint at 500 Hz, and the arm loops through a
reach-and-sweep task. The skin is cast at 15 Hz and a MID-360-like lidar at
10 Hz. A person walks up, reaches into the arm's path, and walks away. The
arm's task speed scales from 100% (nearest skin return at or beyond
`--slow-at`, 0.25 m) to 0 (at or inside `--stop-at`, 0.10 m), so it slows, holds
while the hand is close, and resumes. The robot's own body is told apart by the
geom each ray hit. The sim knows this; the hardware does not, so on the robot
self-returns have to be filtered against the robot model. `--no-react` turns
the reaction off. The docstring has the rest.

The outputs:
- a video: skin returns (dots per sensor), each FOV coloured red (near) to
  green (far), lidar returns (white), and a status banner;
- a plot of skin and lidar distances and the arm's speed;
- an `.npz` with every skin cycle's raw readings, self-hit masks, joint angles
  and speed.

The run shows two properties of HIRO's forearm skin:

- **No sensor looks along the forearm.** Its sensors face out and up from the
  limb. A hand straight ahead of the forearm, in the direction it points, is
  in a blind cone. In the simulation the person reaches in alongside the
  forearm's outer side, where the sensors do look.
- **The lidar sees the approach first.** It has the hand near the arm a
  moment before the skin's sensors pick it up.

## Real skin boards (HIRO firmware) on this PC

HIRO's firmware sends each sensor's frame by Wi-Fi to UDP multicast `239.0.0.1`
on its skin's port: `[1 byte sensor index][64 × uint16 filtered mm][64 × uint16
raw mm]`, little-endian. "Filtered" holds 4000 where the sensor flagged a zone
invalid. To receive it, this PC must be on the boards' Wi-Fi (`HumanoidNetwork`)
and allow inbound UDP on the skin ports through the Windows firewall.

`config/hiro_boards.yaml` says which boards to listen to and which layout sensor
(`<skin>_sensor_<i>`) each firmware index is. A board can drive fewer sensors
than its skin has: the shoulder-yaw board drives 4 of the skin's 6. Edit it by
hand. Find the mapping with `identify`, then set `verified: true`.

```bash
python -m golem_proximity_skin.hiro_view --port 5007                 # 8x8 heatmaps, one board
python -m golem_proximity_skin.hiro_udp status                       # rate + nearest, every board
python -m golem_proximity_skin.hiro_udp identify                     # cover one sensor: which is it?
python -m golem_proximity_skin.hiro_udp record run1.npz --duration 60
python -m golem_proximity_skin.hiro_udp twin --skins right_upper_arm # live readings on the sim H1-2
```

`record` writes the arrays HIRO's recorder does: `timestamps` (ns),
`packets`, `sensor_ids`, `skins` and `skin_indices`. It adds:
- `readings_mm`: decoded, with 0 = no target;
- `layout_sensors`: each frame's layout sensor, -1 if unmapped;
- `ports` and `sources`;
- the layout and board-map YAML in force during the recording.

Zones are in packet order. Their orientation on the physical sensor has not
been checked yet.

**Masking zones the skin obscures.** Where the shell clips a sensor's view, a
few zones read 1–4 mm whatever is in front. `config/hiro_boards.yaml`
`masked_zones` lists them per firmware sensor. `hiro_udp` reports those zones
as no target (raw 0) to every tool, though `record` keeps the packets as
received. The generator applies the same mask to the simulated sensor, so
training data matches the hardware.

To measure the mask from a recording made with the skin mounted and nothing
within half a metre:

```bash
python -m golem_proximity_skin.hiro_udp record open.npz --skins right_upper_arm --duration 30
python -m golem_proximity_skin.hiro_udp mask open.npz --skins right_upper_arm            # dry run
python -m golem_proximity_skin.hiro_udp mask open.npz --skins right_upper_arm --write    # into the board map
```

A zone is obscured if it reads under 6 cm in at least half of the frames where
its sensor mostly sees the room. `--margin N` (default 1) also masks N zones
around each obscured zone: zones beside the shell's edge mix the shell with the
scene and read short. Re-measure whenever the skin is taken apart.

The twin draws the robot at all-zero joints unless you pass `--pose
joint=rad,...`. It only looks right once the board is mounted on that link.

## Sensor model fitted from real recordings

`config/vl53l5cx_model.yaml` describes how the real sensor behaves: its frame
rate, noise against range, the chance a zone is flagged invalid against range,
and how long invalid runs last. It is fitted by
`python -m golem_proximity_skin.sensor_model fit <recordings...>`.

The current fit uses 81,263 frames: HIRO's 9 `raw_data_*` sessions (July–August
2026, `humanoid` branch) plus 8 s from the shoulder-yaw board. Neither needs
ground truth. Noise comes from each zone's frame-to-frame change, with motion
excluded; validity comes from the firmware's own invalid flag.

The fit has no bias term: that needs a flat target at measured distances. Its
under-10 cm numbers likely include the skins' own shell and body returns.
Re-fit as new recordings come in.

The offline simulation and the GOLEM simulator (`GOLEM_TOF_SKIN=1`) apply it by
default. Pass `--ideal-sensor`, or set `GOLEM_TOF_SKIN_IDEAL=1`, for exact
distances.

## Training data

`generate.py` writes randomised, seeded episodes. It runs headless across every
core but one, and needs no ROS. `dataset.py` loads them.

```bash
python -m golem_proximity_skin.generate --out data/arm_v1     --skins right_shoulder right_upper_arm right_forearm --episodes 1000
python -m golem_proximity_skin.generate --out data/knee_v1 --skins right_knee --episodes 500
python -m golem_proximity_skin.dataset summary data/arm_v1
```

Each episode moves the limbs that carry the chosen skins along a random
min-jerk joint trajectory, in a random scene, for 8 s at the fitted sensor
model's rate. The pelvis is fixed and the motion is played back kinematically.
There are three episode types, after ARMOR:
- `avoid`: obstacles tight around the limb's path, never touching it;
- `dynamic`: an obstacle moving in toward the limb, which may make contact;
- `free`: no obstacles.

Each episode also randomises an optional table and wall, sensor mounting error
(±2 mm, ±2°), and the occasional dead sensor.

Per frame and sensor, an episode stores:
- **inputs the hardware also gives:** `readings_mm` and `raw_mm` (from the
  fitted sensor model), and joint `q` and `qd`;
- **labels only the simulation knows:** `ideal_mm`, `hit_class` (none, self,
  static, moving, world), `hit_obstacle`, `nearest_env_mm`, each skin's
  `clearance_m` to every obstacle, sensor poses, and obstacle states.

`generate.py`'s docstring lists every array.

On this 12-core PC it runs at about 11 times real time, about 0.3 MB per
episode. ARMOR's 86.6 h would take about 8 h and 12 GB.

**Limits:**
- The motion is random, not an expert avoiding obstacles. Policy learning still
  needs one, such as retargeted human motion plus a planner.
- All of a frame's sensors are cast at one instant, whereas the firmware reads a
  board's sensors in turn.
- The sensor model has no bias term and uses one ray per zone.

## Baseline perception model

`baseline.py` trains a small CNN on generated episodes. It uses only what the
hardware gives: each sensor's 8×8 readings, raw distances and validity, plus
the joint angles and velocities. From those it predicts, per zone, whether a
return is **none, self or env**, and, per skin, its **clearance** to the
nearest obstacle (capped at 1 m).

```bash
python -m golem_proximity_skin.baseline train data/arm_v1 --out models/arm_v1_baseline
python -m golem_proximity_skin.baseline eval  data/arm_v1 --model models/arm_v1_baseline
```

It's scored on held-out episodes (split by seed: the last 10% test, 10% before
them validation) against rules that need no learning:
- every valid return is environment;
- the best "returns under X mm are self" threshold;
- **the sphere self-filter** (`sphere_filter.py`). This is the method from
  HIRO's `self_filter.py` (branch `nn_test`), applied with GOLEM's 26-sphere
  H1-2 model (`CL_Assets/ros_assets/h1_2_magpie_sphere.urdf`). A valid return
  inside the spheres plus a margin (chosen on validation) is self. The model has
  no leg spheres, so returns from an arm skin onto the legs count as
  environment, and it can't filter for the knee skin;
- for clearance, the skin's nearest reading, with and without perfect
  self-filtering.

`<model>/test_report.json` holds the numbers. It's trained only on simulated
data, so its accuracy on the real skin is unknown until it's checked against
labelled real recordings.

## Preview without ROS

`preview.py` puts the skins on the H1-2 MJCF with a few obstacles. It casts
every sensor with the sim's own caster, prints per-skin hit counts, and renders
the returns and fields of view. It needs only `mujoco`, `numpy`, `pyyaml` and
`pillow`.

```bash
cd core_ws/src/golem_proximity_skin
python3 -m golem_proximity_skin.preview \
    --mjcf ../../../CL_Assets/mujoco_assets/scene_h1_2_magpie.xml \
    --layout config/h12_skin.yaml --out skin_preview.png    # or --viewer
```

## What the model does not do

- Each zone reports the first hit along its centre ray. The real sensor
  reports a histogram-derived distance over the zone's whole cone.
- No noise, no sensitivity to surface reflectance, no ambient light, no
  multipath.
- The sensor's own parent link is excluded. Other robot links are not, so a
  sensor facing the torso or its own hand reads it, as the real skin would.
  Filter self-returns downstream if you need to.

## Tests

```bash
colcon test --packages-select golem_proximity_skin --event-handlers console_direct+
```

`test_layout.py` and `test_vl53l5cx.py` cover the geometry and the zone model.
`test_mujoco_tof.py` casts against a wall at a known distance, and is skipped
without `mujoco`.
