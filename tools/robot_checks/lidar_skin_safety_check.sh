# Read-only check: can the H1-2 run the lidar and the ToF skins together, with the
# GOLEM safety layer, and is the network or the CPU the limit?
#
# Run it from the PC, piped over ssh (nothing is written or installed on the robot):
#   plink -batch -ssh -hostkey <key> -pw <pw> unitree@192.168.123.164 "bash -s -- <label>" < lidar_skin_safety_check.sh
# <label> names the situation, e.g. drivers-wifi, drivers-ethernet, viewer-wifi.
#
# What it does: reads /proc and /sys, lists processes and sockets, listens to ROS
# topics with `timeout`, and listens for the skins' UDP multicast (239.0.0.1,
# ports 5005-5018) with its own sockets. It publishes nothing, binds with
# SO_REUSEADDR so any existing skin receiver keeps getting every packet, and its
# multicast membership ends when it exits. It does not stop or start any node or
# daemon, and changes no setting, driver, config or firmware.
LABEL=${1:-unlabelled}
SKIN_S=${SKIN_S:-30}
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///home/unitree/cyclonedds.xml ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source ~/GOLEM/core_ws/install/setup.bash 2>/dev/null
sec() { echo; echo "=================== $*"; }

sec "label: $LABEL"; date; uptime; echo "cores: $(nproc)"

sec "network interfaces (IPv4) and link state"
ip -4 -o addr show | awk '{print $2, $4}'
for i in $(ls /sys/class/net); do printf "%s state=%s speed=%s rx_drop=%s\n" $i "$(cat /sys/class/net/$i/operstate)" "$(cat /sys/class/net/$i/speed 2>/dev/null)" "$(cat /sys/class/net/$i/statistics/rx_dropped)"; done

sec "who already listens on the skin ports (5005-5018)"
ss -ulpn 2>/dev/null | awk 'NR==1 || $4 ~ /:50(0[5-9]|1[0-8])$/'

sec "running GOLEM pieces"
for n in livox_ros_driver2_node realsense2_camera_node livox_imu_upright safety_node estop_node frame_task_server fastlio_mapping laser_mapping octomap_server self_filter slam_toolbox controller_server locomotion skin; do
  p=$(pgrep -f "$n" | head -1); [ -n "$p" ] && printf "%-24s pid %-7s %s%%\n" "$n" "$p" "$(ps -o pcpu= -p $p)"
done

sec "skins (${SKIN_S} s) while the lidar is measured"
IFACES=$(ip -4 -o addr show | awk '$2!="lo" && $2!~/^docker/ {split($4,a,"/"); print $2"="a[1]}' | tr '\n' ' ')
# The lidar/CPU block runs alongside and prints first (~25 s); the skin summary
# prints when the listen ends. Nothing goes to disk.
{
  sleep 2
  echo "--- CPU during the skin listen (per process, 5 s)"
  top -b -n 2 -d 5 -o %CPU | awk '/^%Cpu/{c++} c==2 && /^%Cpu/{print} c==2 && /^ *[0-9]+ /{print}' | head -14 | cut -c1-120
  echo "--- /livox/lidar"
  timeout 10 ros2 topic hz /livox/lidar 2>&1 | grep -E "average|min" | tail -2
  timeout 8 ros2 topic delay /livox/lidar 2>&1 | grep -E "average|min" | tail -2
} &
LIDAR_PID=$!
timeout $((SKIN_S + 10)) python3 - "$SKIN_S" $IFACES 2>&1 <<'PY'
import collections, selectors, socket, struct, sys, time
dur = float(sys.argv[1])
ifaces = dict(a.split('=') for a in sys.argv[2:])
GROUP = '239.0.0.1'
LABELS = {5005: 'R1 right shoulder', 5006: 'L2 left shoulder', 5007: 'R2 right upper arm',
          5008: 'L3 left upper arm', 5009: 'R3 right forearm', 5010: 'L4 left forearm',
          5011: 'L5 left knee', 5012: 'R5 right knee', 5013: 'L6 left hip', 5014: 'R6 right hip',
          5015: 'H1 head', 5016: 'T1 torso', 5017: 'P1 pelvis R', 5018: 'P2 pelvis L'}
PKTINFO = getattr(socket, 'IP_PKTINFO', 8)
sel = selectors.DefaultSelector()
for port in LABELS:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, 'SO_REUSEPORT'):
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.setsockopt(socket.IPPROTO_IP, PKTINFO, 1)
    s.bind(('', port))
    for name, ip in ifaces.items():
        try:
            s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                         socket.inet_aton(GROUP) + socket.inet_aton(ip))
        except OSError as e:
            print(f'join {GROUP} on {name} failed: {e}')
    s.setblocking(False)
    sel.register(s, selectors.EVENT_READ, port)
t_end = time.time() + dur
seen = collections.defaultdict(list)            # (port, chip) -> arrival times
size = collections.Counter(); src = collections.defaultdict(set); via = collections.defaultdict(set)
while time.time() < t_end:
    for key, _ in sel.select(timeout=0.2):
        while True:
            try:
                data, anc, _, addr = key.fileobj.recvmsg(2048, 64)
            except BlockingIOError:
                break
            now = time.time()
            port = key.data
            chip = data[0] if data else -1
            seen[(port, chip)].append(now)
            size[len(data)] += 1
            src[port].add(addr[0])
            for lvl, typ, val in anc:
                if lvl == socket.IPPROTO_IP and typ == PKTINFO:
                    try:
                        via[port].add(socket.if_indextoname(struct.unpack('i', val[:4])[0]))
                    except OSError:
                        pass
total = sum(len(v) for v in seen.values())
print(f'skin packets: {total} in {dur:.0f} s ({total / dur:.0f}/s); sizes {dict(size.most_common(3))}')
if not total:
    print('NO SKIN PACKETS: boards off, on another network, or the multicast is not reaching this PC')
for port, lab in LABELS.items():
    chips = sorted(c for (p, c) in seen if p == port)
    if not chips:
        continue
    rates, gaps = [], []
    for c in chips:
        t = seen[(port, c)]
        if len(t) > 2:
            rates.append((len(t) - 1) / (t[-1] - t[0]))
            gaps.append(max(b - a for a, b in zip(t, t[1:])))
    print(f'{port} {lab:20} from {",".join(sorted(src[port]))} via {",".join(sorted(via[port])) or "?"}: '
          f'{len(chips)} chips {chips}, Hz per chip {min(rates):.1f}-{max(rates):.1f}, '
          f'worst gap {1000 * max(gaps):.0f} ms' if rates else f'{port} {lab}: {len(chips)} chips, too few packets')
PY
wait $LIDAR_PID

sec "GOLEM safety layer"
p=$(pgrep -f "h12_safety_layer|safety_node" | head -1)
if [ -z "$p" ]; then
  echo "safety_node is not running (start h12_control to check it)"
else
  echo "safety_node pid $p, $(ps -o pcpu=,etime= -p $p) (%CPU, run time); busiest threads:"
  top -H -b -n 1 -p $p | sed -n '8,12p' | cut -c1-100
  echo "--- /lowcmd (safety_node's 500 Hz relay; its only writer should be safety_node)"
  timeout 10 ros2 topic info -v /lowcmd --no-daemon 2>&1 | grep -E "Publisher count|Subscription count|Node name" | head -6
  timeout 8 ros2 topic hz /lowcmd 2>&1 | grep -E "average|min" | tail -2
  echo "--- /lowstate"
  timeout 8 ros2 topic hz /lowstate 2>&1 | grep -E "average|min" | tail -2
  echo "--- upper-body input (who writes rt/safety/lowcmd_upper_in)"
  timeout 10 ros2 topic info -v /safety/lowcmd_upper_in --no-daemon 2>&1 | grep -E "Publisher count|Node name" | head -4
  echo "--- hardware estop status"
  timeout 6 ros2 topic hz /h12/estop_status_raw 2>&1 | grep -E "average" | tail -1
fi

sec "temperature and throttling"
for z in /sys/class/thermal/thermal_zone*; do printf "%s=%sC " $(cat $z/type) $(( $(cat $z/temp)/1000 )); done; echo
for c in 0 4 11; do printf "cpu$c throttle: "; cat /sys/devices/system/cpu/cpu$c/thermal_throttle/core_throttle_count 2>/dev/null; done
sec "done ($LABEL)"
