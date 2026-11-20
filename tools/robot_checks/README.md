# Lidar + skin + safety layer check (read-only)

`lidar_skin_safety_check.sh` shows whether the H1-2 can run the lidar and the ToF
skins at the same time under the GOLEM safety layer, and whether the network
or the CPU is the limit. It takes about 1.5 min per run.

## What it does on the robot

- **Only reads and listens:**
  - it reads `/proc` and `/sys`, and lists processes and sockets;
  - it listens to ROS topics, each under `timeout`;
  - it listens for the skins' multicast (239.0.0.1, ports 5005–5018) on its own
    sockets. These use `SO_REUSEADDR`, so a skin receiver that is already
    running still gets every packet.
- **Changes nothing:** it publishes nothing, writes no files, starts or stops
  nothing, and changes no setting, driver, config or firmware.
- **The one side effect:** `ros2 topic hz /lowcmd` subscribes to the 500 Hz
  command stream for 8 s, which costs part of one core while it runs.

## Before you run it

Start what you'll actually use, natively on the robot, as GOLEM's rules say for the H1-2:

```
ros2 launch h12_bringup h12_real_drivers.launch.py use_grippers:=false
# and, for the safety-layer part, the control block with the e-stop box connected:
ros2 launch h12_bringup h12_control.launch.py use_estop:=true use_frame_task:=false
```

Power the skin boards.

## Runs, from the laptop

Run these from `tools/robot_checks/`, with the robot password from your own notes:

```
plink -batch -ssh -hostkey SHA256:9sAfqxQxTLI19yIsVX6/3DYqkRryyrxLAnO4oMJepyo -pw <pw> ^
  unitree@192.168.123.164 "bash -s -- drivers-wifi" < lidar_skin_safety_check.sh > check_drivers-wifi.txt
```

| Label | Laptop | Viewer |
|---|---|---|
| `drivers-wifi` | on Wi-Fi | none |
| `viewer-wifi` | on Wi-Fi | full lidar view open (Nick's viewer, or `lidar_view.py --pts 7000`) |
| `viewer-ethernet` | on Ethernet to the robot's router or switch | same viewer |

## Reading the result

- **The `skins` block:**
  - **Hz per chip** should be about 12–15 for every live chip.
  - **A worst gap** over about 200 ms means lost frames.
  - **"NO SKIN PACKETS"**, while the boards are on, means the multicast isn't
    reaching the robot's wired port. That decides where the skin receiver can
    run.
- **The `/livox/lidar` block:** it should stay at 30 Hz with a steady delay in
  every run.
- **The safety block:**
  - `/lowcmd` should be about 500 Hz;
  - its publisher should be the safety layer only;
  - `/safety/lowcmd_upper_in` should have no writer, or exactly one.
- **Comparing runs:**
  - If skin Hz or gaps get worse from `drivers-wifi` to `viewer-wifi` and recover
    on `viewer-ethernet`, Wi-Fi airtime is the limit, and Ethernet is the fix.
  - If the robot's CPU idle drops a lot or `/lowcmd` falls below 500 Hz, CPU is
    the limit.
