"""The H1-2's proximity skins: frames and visuals, point clouds, and the lidar + skin monitor.

    ros2 launch golem_proximity_skin skin.launch.py                 # with the sim (GOLEM_TOF_SKIN=1)
    ros2 launch golem_proximity_skin skin.launch.py use_sim_time:=false use_rviz:=false

Run beside a bringup (h12_sim.launch.py), which provides robot_state_publisher
and the robot's TF; this file starts only the skin's nodes and, with use_rviz,
an RViz of the robot, the lidar and the skins.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('golem_proximity_skin')
    params = [{
        'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
        'layout': LaunchConfiguration('layout'),
    }]

    def node(executable):
        return Node(package='golem_proximity_skin', executable=executable, name=executable,
                    parameters=params, output='screen', emulate_tty=True)

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('layout', default_value=os.path.join(share, 'config', 'h12_skin.yaml')),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=os.path.join(share, 'rviz', 'skin.rviz')),
        node('skin_description'),
        node('skin_cloud'),
        node('skin_monitor'),
        Node(
            package='rviz2', executable='rviz2', name='rviz2_skin',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': ParameterValue(
                LaunchConfiguration('use_sim_time'), value_type=bool)}],
            output='screen', emulate_tty=True,
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
    ])
