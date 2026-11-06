"""Subscribe to the H1-2's lidar and its proximity skins, and keep the latest of each.

    from golem_proximity_skin.listener import SensorListener
    listener = SensorListener(node, layout)
    listener.on_skin(lambda name, readings: ...)      # (n_sensors, 64) uint16 mm
    listener.on_lidar(lambda frame_id, xyz: ...)      # (n, 3) float32, frame_id's frame
    listener.latest_skin('left_forearm'), listener.latest_lidar()

Works the same against the sim and the robot: it reads only topics both
publish. Subscriptions use sensor-data QoS (best effort), which matches both
reliable and best-effort publishers.
"""
import threading
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import UInt16MultiArray

from . import messages
from .layout import SkinLayout

LIDAR_TOPIC = '/livox/pointcloud'

_PF_DTYPES = {
    PointField.INT8: np.int8, PointField.UINT8: np.uint8,
    PointField.INT16: np.int16, PointField.UINT16: np.uint16,
    PointField.INT32: np.int32, PointField.UINT32: np.uint32,
    PointField.FLOAT32: np.float32, PointField.FLOAT64: np.float64,
}


def cloud_xyz(msg: PointCloud2) -> np.ndarray:
    """(n, 3) float32 x/y/z of a PointCloud2, whatever its point layout; NaNs dropped."""
    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in 'xyz'):
        raise ValueError('PointCloud2 has no x/y/z fields')
    dtype = np.dtype({
        'names': list('xyz'),
        'formats': [np.dtype(_PF_DTYPES[fields[k].datatype]).newbyteorder(
            '>' if msg.is_bigendian else '<') for k in 'xyz'],
        'offsets': [fields[k].offset for k in 'xyz'],
        'itemsize': msg.point_step,
    })
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.row_step and msg.height > 1 and msg.row_step != msg.point_step * msg.width:
        buf = buf.reshape(msg.height, msg.row_step)[:, :msg.point_step * msg.width].ravel()
    arr = np.frombuffer(buf.tobytes(), dtype=dtype, count=n)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float32)
    return xyz[np.isfinite(xyz).all(axis=1)]


class SensorListener:
    def __init__(self, node, layout: SkinLayout, lidar_topic: str = LIDAR_TOPIC):
        self.node = node
        self.layout = layout
        self._lock = threading.Lock()
        self._skin: Dict[str, Tuple[float, np.ndarray]] = {}
        self._lidar: Optional[Tuple[float, str, np.ndarray]] = None
        self._skin_cbs = []
        self._lidar_cbs = []
        self._expected = {s.name: len(s.sensors) for s in layout.enabled_skins}
        self._subs = [node.create_subscription(
            UInt16MultiArray, messages.tof_topic(name),
            lambda msg, name=name: self._on_tof(name, msg), qos_profile_sensor_data)
            for name in self._expected]
        self._subs.append(node.create_subscription(
            PointCloud2, lidar_topic, self._on_lidar, qos_profile_sensor_data))

    def on_skin(self, cb: Callable[[str, np.ndarray], None]) -> None:
        self._skin_cbs.append(cb)

    def on_lidar(self, cb: Callable[[str, np.ndarray], None]) -> None:
        self._lidar_cbs.append(cb)

    def latest_skin(self, name: str) -> Optional[np.ndarray]:
        with self._lock:
            entry = self._skin.get(name)
        return None if entry is None else entry[1]

    def skin_age(self, name: str) -> float:
        """Wall seconds since the skin's last message (inf before the first)."""
        with self._lock:
            entry = self._skin.get(name)
        return float('inf') if entry is None else time.monotonic() - entry[0]

    def latest_lidar(self) -> Optional[Tuple[str, np.ndarray]]:
        with self._lock:
            entry = self._lidar
        return None if entry is None else (entry[1], entry[2])

    def lidar_age(self) -> float:
        with self._lock:
            entry = self._lidar
        return float('inf') if entry is None else time.monotonic() - entry[0]

    def _on_tof(self, name: str, msg: UInt16MultiArray) -> None:
        try:
            readings = messages.from_msg(msg)
        except ValueError as e:
            self.node.get_logger().warn(f'{messages.tof_topic(name)}: {e}', throttle_duration_sec=5.0)
            return
        if readings.shape[0] != self._expected[name]:
            self.node.get_logger().warn(
                f'{messages.tof_topic(name)}: {readings.shape[0]} sensors, layout has '
                f'{self._expected[name]}', throttle_duration_sec=5.0)
            return
        with self._lock:
            self._skin[name] = (time.monotonic(), readings)
        for cb in self._skin_cbs:
            cb(name, readings)

    def _on_lidar(self, msg: PointCloud2) -> None:
        try:
            xyz = cloud_xyz(msg)
        except (ValueError, KeyError) as e:
            self.node.get_logger().warn(f'lidar cloud: {e}', throttle_duration_sec=5.0)
            return
        with self._lock:
            self._lidar = (time.monotonic(), msg.header.frame_id, xyz)
        for cb in self._lidar_cbs:
            cb(msg.header.frame_id, xyz)
