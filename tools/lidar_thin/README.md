# lidar_thin: H1-2 lidar view over Wi-Fi, thinned on the robot

**This is a viewing workaround, not part of the control stack.** It lets you
watch the H1-2's Livox MID-360 cloud, alone or with Nick's lidar + ToF-skin
viewer, from a PC on Wi-Fi without the lag.

It sends a smaller copy of each scan, and it changes nothing on the robot.
Nothing that steers, plans or keeps the robot safe should rely on it. See
[Restrictions](#restrictions) before using it for anything beyond looking.

## Why it exists

Measured on the robot on 2026-11-16, with GOLEM's `h12_real_drivers.launch.py`
running:

| Measured | Result |
|---|---|
| `/livox/lidar` on the robot | 30.0 Hz, 41 ms from scan to topic, steady. 92–96% of the 12 cores idle. |
| One full scan | about 6,700 points, 0.13 MB |
| Full scans at 30 Hz | about 4 MB/s, or 32 Mbit/s |
| The lab Wi-Fi (HumanoidNetwork) | 2.4 GHz, channel 7, overlapping busy neighbouring networks |

- **The robot keeps up; the network doesn't.** The lag seen on a PC comes from
  pulling every full scan over that Wi-Fi.
- **Thinned on the robot to about 0.7 MB/s,** the same link added 24 ms on
  average (95% of frames under 58 ms). No frame came more than 195 ms after
  the one before.
- **The ToF skins share that 2.4 GHz air.** They are ESP32 boards sending Wi-Fi
  multicast, so a full lidar stream to a Wi-Fi PC also costs the skins airtime.

**The better fix is an Ethernet cable** from the PC to the robot's router or
switch. Use this tool when you have to stay on Wi-Fi.

## Where it sits in the stack

```
 robot onboard PC (native, ROS 2 Humble, ROS_DOMAIN_ID=0)          PC (any OS)
 ────────────────────────────────────────────────────────          ──────────────────────────
 livox_ros_driver2 ── /livox/lidar (CustomMsg, 30 Hz) ──┬──► FAST-LIO, self-filter, ...   (unchanged, full scans)
                                                        │
 Unitree firmware ─── /lowstate (500 Hz) ───────────────┤
                                                        │
 h12_safety_layer ─── rt/lowcmd (500 Hz) ◄─ rt/safety/* │     (untouched: the tap never
                                                        │      reads or writes these)
                                                        ▼
                                 lidar_tap.py  (temporary, read-only, best-effort,
                                 started over ssh, from memory, capped by `timeout`)
                                   thin each scan ─► newest frame only ─► ssh stdout
                                                                    │  Wi-Fi or Ethernet
                                                                    ▼
                                              A. lidar_view.py   (numpy + matplotlib window)
                                              B. thin_relay.py ─► 127.0.0.1:7420 ─► Nick's main.py --relay
```

**Compared with Nick's stock path** (`proximity_point_motion`, branch `humanoid-2`):

| | Stock | Here |
|---|---|---|
| Robot | `h12_real_drivers.launch.py` | the same |
| PC side | `scripts/start_golem_ros.sh` (GOLEM's `docker_run.sh ros` on domain 0), then `scripts/run_lidar_tof.sh` | `thin_relay.py` replaces both scripts and needs no Docker |
| What crosses the link | every full scan, from a `golem_ros` container that joins the robot's ROS network over the link | thinned scans and 50 Hz joint states, down one ssh connection |
| Viewer | `main.py`, unchanged | `main.py`, unchanged; it reads the same relay socket format (`sensor_relay.py`'s framing) |

The PC never joins the robot's DDS domain.

## Requirements

**Robot:** GOLEM's workspace built at `~/GOLEM/core_ws`, which provides
`livox_ros_driver2` and, for mode B, `unitree_hg`. It also needs ROS 2 Humble,
python3 with numpy, `/home/unitree/cyclonedds.xml`, and the drivers running:

```
ros2 launch h12_bringup h12_real_drivers.launch.py use_grippers:=false
```

**PC:** Python 3.10+ and numpy, and a way to log in:
- **Windows:** plink (PuTTY), with the password in `H12_PW` or typed at the prompt.
- **Linux:** OpenSSH with keys, or `SSHPASS` with sshpass.

Each mode then needs:
- **Mode A:** matplotlib.
- **Mode B:** Nick's `proximity_point_motion` on `humanoid-2`, and a local ROS 2
  so `main.py` can decode messages.

The robot address and host key default to the lab H1-2
(`unitree@192.168.123.164`). Override them with `H12_HOST` / `H12_HOSTKEY` or `--host`.

## Use

Run these from `tools/lidar_thin/`.

### A. Quick view, on any PC including Windows

```
set H12_PW=<robot password>                                   # Windows; or you are prompted
python lidar_view.py                                          # 4000 pts/frame, every frame
python lidar_view.py --radius 3 --pts 2000                    # the robot's surroundings only
python lidar_view.py --hz 10                                  # a very bad link
```

The window has two panels:
- **Top-down** and **side**, coloured by height.
- **Keys:** `f` flips the cloud, space pauses.

The status line splits the lag in two:
- **robot:** rate, and the delay from scan time to the tap. This shows whether
  the robot is keeping up.
- **laptop:** rate, extra delay on the link, worst gap, and frames skipped.
  This is what the network costs.

Every frame's timing goes to `lidar_view_timing.csv`, which git ignores.

### B. Nick's lidar + skin viewer, thinned

```
# terminal 1
python3 thin_relay.py --radius 4 --pts 3000
# terminal 2, in proximity_point_motion (humanoid-2)
python3 main.py --lidar --lidar-topic /livox/points --live-joints \
    --joint-topic /joint_states --relay 127.0.0.1:7420 --no-camera --no-record
```

### Thinning flags, for both modes

| Flag | Default | Effect |
|---|---|---|
| `--pts` | 4000 (A) / 3000 (B) | Points sent per scan. A full scan is about 6,700. |
| `--radius` | off | Drop points farther than this from the lidar, in metres. |
| `--near` | 1.5 m | Points this close are always kept first; the rest are a random sample. |
| `--hz` | every scan | Maximum scans per second. |
| `--minutes` | 30 | Hard cap on the robot-side process. |
| `--joint-hz` (B) | 50 | `/joint_states` built from `/lowstate`; 0 sends lidar only. |

## What it does and doesn't do on the robot

**It does:**
- Start one `python3` process through ssh. The code comes from ssh stdin, so
  no file is written on the robot.
- Make that process a ROS node named `thin_tap_readonly`, with no rosout and no
  parameter services.
- Subscribe it to `/livox/lidar`, and in mode B to `/lowstate`. Both
  subscriptions are best-effort with a queue of 1, so the drivers never wait
  for it.
- Cost about 6% of one core while it runs. The extra copy of each scan stays in
  the robot's memory.
- Exit when the PC side closes (a broken pipe) and in any case at `--minutes`.

**It doesn't:**
- Install, configure, restart or rebuild anything.
- Change any driver setting (lidar rate, point count) or launch file.
- Publish anything on the robot's ROS network.
- Touch the GOLEM safety layer, `rt/lowcmd`, `rt/safety/*` or the e-stop.
- Change what other subscribers get: FAST-LIO, the self-filter and octomap
  still receive full scans.

## Restrictions

1. **Viewing only.**
   - **Thinned:** points beyond `--near` are a random sample, so distant points
     flicker from scan to scan.
   - **Lower precision:** mode A sends float16, about 1 cm at close range.
   - **Frames can be skipped:** each stream sends only its newest frame, so on a
     slow link frames are dropped rather than queued.
   - **Slower joints:** mode B's joint states are 50 Hz, not 500 Hz.

   Don't feed this stream to collision avoidance, planning or any safety check.
   Policies and safety logic should run on the robot and read the full topics.
2. **Not for recording datasets.** If Nick's viewer records through the relay,
   it saves thinned clouds and 50 Hz joints. Record on the robot, or over
   Ethernet with the full topics.
3. **It hides robot-side processing.** The robot delay it reports is for the
   raw `/livox/lidar` only. It says nothing about FAST-LIO, the self-filter,
   octomap or Nav2. If the full bringup is slow, this view won't show it.
4. **It runs code on the robot.** It's temporary and read-only, but it is still
   a process on a shared lab robot. Coordinate with whoever is running the
   robot, as GOLEM's `CLAUDE.md` asks for the real robot ("coordinate before
   touching"). Each open viewer starts its own tap; open one at a time.
5. **The ROS domain.** The tap is a node on `ROS_DOMAIN_ID=0`, the real robot's
   command bus. It must stay listen-only:
   - don't change it to publish;
   - don't run a simulator against the same domain (GOLEM: "one robot stack per
     DDS domain").
6. **It depends on outside message formats.**
   - **Livox CustomMsg layout:** the tap reads the point array straight from the
     message bytes. It checks the array length against the message size and
     refuses frames that don't match, rather than misreading them. GOLEM pins
     `livox_ros_driver2` at `13eb05e` and forbids bumping it; if that ever
     changes, re-run the tests on a real frame.
   - **Mode B, `sensor_relay.py`'s socket format:** mode B reproduces it (u16
     topic length, u32 payload length, u64 receive ns). If Nick changes that
     format, mode B breaks.
   - **Mode B, joint order:** it uses Nick's 27-joint H1-2 order for `/lowstate`.
   - **The robot's software:** checked against ROS 2 Humble and GOLEM commit
     `3662665` (branch `camera_ready`, 2026-10-28). The newer C++
     `golem_safety` doesn't affect it, since it reads only `/livox/lidar` and
     `/lowstate`.
7. **Credentials.**
   - **The password is never stored.** It comes from `H12_PW`, a prompt, or
     `SSHPASS`.
   - **plink shows it to local users:** plink takes it as `-pw`, so it appears
     in the PC's process list. ssh keys avoid that.
   - **The lab robot's address, user name and host-key fingerprint are built in**
     as defaults. They're not secrets, but remove them (they're overridable)
     before putting this in a public repo.
8. **Licence:** none has been chosen. Treat it like the rest of this repo:
   collaborators only.

## Status

| Part | Status |
|---|---|
| Mode A, first version (no `--radius` / `--near` / `--hz`) | Run on the robot, 2026-11-16: 6,071 frames over 202 s, at the numbers above. |
| Thinning, CustomMsg parsing, relay framing, latest-only queueing | Covered by offline tests. |
| New flags on real scans | Not yet run on the robot. |
| Mode B with Nick's `main.py` | Not yet run. Its PointCloud2 and JointState messages are built with rclpy on the robot. |
| `unitree_hg` importable on the robot | Not checked. Without it, mode B says so and sends lidar only. |

## Files

| File | Runs on | What it does |
|---|---|---|
| `lidar_tap.py` | robot (via ssh stdin) | Read-only listener: parse, thin, newest-only, write frames to stdout. |
| `robot_link.py` | PC | Starts the tap over plink or ssh with the settings; shared by both modes. |
| `lidar_view.py` | PC | Mode A viewer and timing log. |
| `thin_relay.py` | PC | Mode B: serves the tap's frames to Nick's `main.py --relay`. |
| `test_lidar_thin.py` | PC | Offline tests: `python -m pytest tools/lidar_thin -q`. No robot or ROS needed. |

The read-only check that measures lidar, skins and the safety layer together
is in `../robot_checks/`.
