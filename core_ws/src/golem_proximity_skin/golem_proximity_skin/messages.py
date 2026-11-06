"""The ToF skin's wire format, shared by the sim, the hardware bridge and the nodes.

Each skin publishes one std_msgs/UInt16MultiArray per ranging cycle on
tof_topic(name), the message type proximity_point_motion records its raw
readings as: dim[0] 'sensor' (sensors in layout order), dim[1] 'zone'
(vl53l5cx zone order), data in mm with vl53l5cx.NO_TARGET for no return.
"""
import numpy as np
from std_msgs.msg import MultiArrayDimension, UInt16MultiArray

from . import vl53l5cx

TOPIC_PREFIX = '/skin'


def tof_topic(skin_name: str) -> str:
    return f'{TOPIC_PREFIX}/{skin_name}/tof'


def points_topic(skin_name: str) -> str:
    return f'{TOPIC_PREFIX}/{skin_name}/points'


def to_msg(readings: np.ndarray) -> UInt16MultiArray:
    r = np.ascontiguousarray(readings, dtype=np.uint16)
    n, z = r.shape
    msg = UInt16MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(label='sensor', size=n, stride=n * z),
        MultiArrayDimension(label='zone', size=z, stride=z),
    ]
    msg.data = r.ravel().tolist()
    return msg


def from_msg(msg: UInt16MultiArray) -> np.ndarray:
    """(n_sensors, NUM_ZONES) readings; raises ValueError on a malformed message."""
    data = np.asarray(msg.data, dtype=np.uint16)
    z = vl53l5cx.NUM_ZONES
    if data.size % z:
        raise ValueError(f'{data.size} readings is not a multiple of {z} zones')
    return data.reshape(-1, z)
