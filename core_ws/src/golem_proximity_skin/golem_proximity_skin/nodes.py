"""The skin's ROS nodes. Each takes the `layout` parameter (a layout YAML path;
default: this package's config/h12_skin.yaml).

skin_description  /tf_static: <parent> -> <name>_skin -> <name>_sensor_<i>, and
                  /skin/description_markers (latched): the skin meshes (or sleeves) and
                  sensor boards, frame-locked so they ride the robot in RViz.
skin_cloud        /skin/<name>/tof -> /skin/<name>/points (PointCloud2 in
                  <name>_skin; fields x y z range sensor zone, all float32).
skin_monitor      subscribes to every skin and the lidar (listener.SensorListener);
                  publishes /skin/proximity_markers (each sensor's field of view,
                  cut at its nearest return and coloured by it) and
                  /skin/nearest_range (Float32, m, the nearest return of any
                  sensor; +inf when none), and logs both sensors' status.
"""
import math
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Float32, UInt16MultiArray
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from . import messages, vl53l5cx
from .layout import load_layout, matrix_to_quat_xyzw
from .listener import SensorListener

SKIN_COLOR = (0.25, 0.7, 0.9, 0.35)      # proximity_point_motion's dermis colour
BOARD_COLOR = (0.1, 0.1, 0.12, 1.0)
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def default_layout() -> str:
    return os.path.join(get_package_share_directory('golem_proximity_skin'), 'config', 'h12_skin.yaml')


def _layout(node: Node):
    path = node.declare_parameter('layout', '').value or default_layout()
    layout = load_layout(path)
    node.get_logger().info(
        f'layout {path}: {len(layout.enabled_skins)} skins, {layout.sensor_count()} sensors')
    return layout


def _color(rgba) -> ColorRGBA:
    return ColorRGBA(r=float(rgba[0]), g=float(rgba[1]), b=float(rgba[2]), a=float(rgba[3]))


def _set_pose(msg_pose, pose) -> None:
    msg_pose.position.x, msg_pose.position.y, msg_pose.position.z = (float(v) for v in pose.xyz)
    q = matrix_to_quat_xyzw(pose.R)
    msg_pose.orientation.x, msg_pose.orientation.y, msg_pose.orientation.z, msg_pose.orientation.w = q


def _identity(msg_pose) -> None:
    msg_pose.orientation.w = 1.0


class SkinDescription(Node):
    def __init__(self):
        super().__init__('skin_description')
        layout = _layout(self)
        self._tf = StaticTransformBroadcaster(self)
        stamp = self.get_clock().now().to_msg()
        tfs = []
        for skin in layout.enabled_skins:
            tfs.append(self._edge(stamp, skin.parent, skin.frame, skin.origin))
            tfs += [self._edge(stamp, skin.frame, skin.sensor_frame(i), p)
                    for i, p in enumerate(skin.sensors)]
        self._tf.sendTransform(tfs)
        self._pub = self.create_publisher(MarkerArray, '/skin/description_markers', LATCHED)
        self._pub.publish(self._markers(layout, stamp))
        self.get_logger().info(f'published {len(tfs)} skin frames')

    @staticmethod
    def _edge(stamp, parent, child, pose) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (
            float(v) for v in pose.xyz)
        q = matrix_to_quat_xyzw(pose.R)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        return t

    @staticmethod
    def _markers(layout, stamp) -> MarkerArray:
        out = MarkerArray()
        mid = 0
        for skin in layout.enabled_skins:
            m = Marker()
            m.header.frame_id = skin.frame
            m.header.stamp = stamp
            m.ns, m.id = 'dermis', mid
            m.frame_locked = True
            m.action = Marker.ADD
            _identity(m.pose)
            m.color = _color(SKIN_COLOR)
            if skin.mesh:
                m.type = Marker.MESH_RESOURCE
                m.mesh_resource = skin.mesh
                m.scale.x = m.scale.y = m.scale.z = 1.0
            elif skin.radius is not None:
                # A CYLINDER marker's axis is its z, the sleeve's limb axis.
                m.type = Marker.CYLINDER
                m.scale.x = m.scale.y = 2.0 * (skin.radius - 0.004)
                m.scale.z = skin.length
            else:
                continue
            out.markers.append(m)
            mid += 1
        for skin in layout.enabled_skins:
            for i in range(len(skin.sensors)):
                m = Marker()
                m.header.frame_id = skin.sensor_frame(i)
                m.header.stamp = stamp
                m.ns, m.id = 'sensor_board', mid
                m.frame_locked = True
                m.type = Marker.CUBE
                _identity(m.pose)
                m.pose.position.z = -0.002
                m.scale.x, m.scale.y, m.scale.z = 0.016, 0.012, 0.004
                m.color = _color(BOARD_COLOR)
                out.markers.append(m)
                mid += 1
        return out


CLOUD_FIELDS = [PointField(name=n, offset=4 * k, datatype=PointField.FLOAT32, count=1)
                for k, n in enumerate(('x', 'y', 'z', 'range', 'sensor', 'zone'))]


class SkinCloud(Node):
    def __init__(self):
        super().__init__('skin_cloud')
        layout = _layout(self)
        self._skins = {s.name: s for s in layout.enabled_skins}
        self._pubs = {n: self.create_publisher(PointCloud2, messages.points_topic(n), qos_profile_sensor_data)
                      for n in self._skins}
        self._subs = [self.create_subscription(
            UInt16MultiArray, messages.tof_topic(n),
            lambda msg, n=n: self._on_tof(n, msg), qos_profile_sensor_data) for n in self._skins]

    def _on_tof(self, name, msg) -> None:
        skin = self._skins[name]
        try:
            readings = messages.from_msg(msg)
        except ValueError as e:
            self.get_logger().warn(f'{name}: {e}', throttle_duration_sec=5.0)
            return
        rows = []
        for i, (pose, r) in enumerate(zip(skin.sensors, readings)):
            pts, zones = vl53l5cx.readings_to_points(r)
            if not len(pts):
                continue
            rng = np.linalg.norm(pts, axis=1)
            rows.append(np.column_stack([pts @ pose.R.T + pose.xyz, rng,
                                         np.full(len(pts), i), zones]))
        data = (np.vstack(rows) if rows else np.zeros((0, 6))).astype(np.float32)
        out = PointCloud2()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = skin.frame
        out.height, out.width = 1, data.shape[0]
        out.fields = CLOUD_FIELDS
        out.is_bigendian = False
        out.point_step = 24
        out.row_step = 24 * data.shape[0]
        out.is_dense = True
        out.data = data.tobytes()
        self._pubs[name].publish(out)


def _proximity_rgba(d: float, near: float, far: float):
    """Red at `near` and closer, through yellow, to green at `far`; grey for no return."""
    if not math.isfinite(d):
        return (0.6, 0.6, 0.6, 0.25)
    t = min(max((d - near) / (far - near), 0.0), 1.0)
    return (min(1.0, 2.0 * (1.0 - t)), min(1.0, 2.0 * t), 0.1, 0.8)


class SkinMonitor(Node):
    def __init__(self):
        super().__init__('skin_monitor')
        self.layout = _layout(self)
        self.near = float(self.declare_parameter('near_m', 0.10).value)
        self.far = float(self.declare_parameter('far_m', 0.50).value)
        self.idle_length = float(self.declare_parameter('idle_fov_length_m', 0.08).value)
        self.listener = SensorListener(self, self.layout,
                                       self.declare_parameter('lidar_topic', '/livox/pointcloud').value)
        self.listener.on_skin(self._on_skin)
        self._marker_pub = self.create_publisher(MarkerArray, '/skin/proximity_markers', 10)
        self._nearest_pub = self.create_publisher(Float32, '/skin/nearest_range', 10)
        self._nearest = {}     # skin -> per-sensor nearest return (m, inf for none)
        half = math.radians(vl53l5cx.FOV_DEG / 2.0)
        self._corners = np.array([[math.tan(sx * half), math.tan(sy * half), 1.0]
                                  for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
        self.create_timer(2.0, self._report)

    def _on_skin(self, name, readings) -> None:
        r = readings.astype(np.float64) / 1000.0
        r[readings == vl53l5cx.NO_TARGET] = np.inf
        nearest = r.min(axis=1)
        self._nearest[name] = nearest
        skin = next(s for s in self.layout.enabled_skins if s.name == name)
        out = MarkerArray()
        for i, d in enumerate(nearest):
            m = Marker()
            m.header.frame_id = skin.sensor_frame(i)
            m.ns, m.id = name, i
            m.frame_locked = True
            m.type = Marker.LINE_LIST
            m.lifetime = Duration(sec=1)
            _identity(m.pose)
            m.scale.x = 0.003
            m.color = _color(_proximity_rgba(float(d), self.near, self.far))
            depth = float(d) if math.isfinite(d) else self.idle_length
            c = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in self._corners * depth]
            o = Point()
            for k in range(4):
                m.points += [o, c[k], c[k], c[(k + 1) % 4]]
            out.markers.append(m)
        self._marker_pub.publish(out)
        overall = min((float(v.min()) for v in self._nearest.values()), default=math.inf)
        self._nearest_pub.publish(Float32(data=overall))

    def _report(self) -> None:
        log = self.get_logger()
        lidar = self.listener.latest_lidar()
        if lidar is None or self.listener.lidar_age() > 2.0:
            lidar_s = 'lidar: no data'
        else:
            frame, xyz = lidar
            rng = np.linalg.norm(xyz, axis=1) if len(xyz) else np.zeros(0)
            lidar_s = (f'lidar: {len(xyz)} pts in {frame}, nearest '
                       f'{rng.min():.2f} m' if len(rng) else f'lidar: 0 pts in {frame}')
        stale = [s.name for s in self.layout.enabled_skins if self.listener.skin_age(s.name) > 2.0]
        best = (math.inf, None)
        for name, nearest in self._nearest.items():
            i = int(np.argmin(nearest))
            if nearest[i] < best[0]:
                best = (float(nearest[i]), f'{name}_sensor_{i}')
        skin_s = (f'skin: nearest {best[0]:.3f} m at {best[1]}' if best[1] and math.isfinite(best[0])
                  else 'skin: no returns')
        if stale:
            skin_s += f' | no data from {", ".join(stale)}'
        log.info(f'{lidar_s} | {skin_s}')


def _spin(cls):
    rclpy.init()
    node = cls()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def description_main():
    _spin(SkinDescription)


def cloud_main():
    _spin(SkinCloud)


def monitor_main():
    _spin(SkinMonitor)
