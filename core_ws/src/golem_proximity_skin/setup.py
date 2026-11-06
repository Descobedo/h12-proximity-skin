import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'golem_proximity_skin'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'meshes', 'skin'), glob('meshes/skin/*.stl')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='diego',
    maintainer_email='descobed@bu.edu',
    description='VL53L5CX ToF proximity skins on the H1-2.',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'skin_description = golem_proximity_skin.nodes:description_main',
            'skin_cloud = golem_proximity_skin.nodes:cloud_main',
            'skin_monitor = golem_proximity_skin.nodes:monitor_main',
            'skin_preview = golem_proximity_skin.preview:main',
            'skin_simulate = golem_proximity_skin.simulate:main',
            'skin_generate = golem_proximity_skin.generate:main',
            'skin_dataset = golem_proximity_skin.dataset:main',
            'skin_sensor_model = golem_proximity_skin.sensor_model:main',
            'skin_hiro_udp = golem_proximity_skin.hiro_udp:main',
            'skin_hiro_view = golem_proximity_skin.hiro_view:main',
        ],
    },
)
