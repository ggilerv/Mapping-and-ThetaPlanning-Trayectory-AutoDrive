import os
from glob import glob

from setuptools import setup, find_packages

package_name = 'global_planner'

setup(
    name=package_name,
    version='0.1.0',
    # find_packages() also picks up the vendored python_motion_planning
    # library (global_planner.python_motion_planning and its subpackages),
    # so it gets installed alongside this package with no external
    # dependency on any other repo being present on the machine.
    packages=find_packages(include=[package_name, f'{package_name}.*'], exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        (os.path.join('share', package_name, 'waypoints'), glob('waypoints/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='George Gabriel Giler Vega',
    maintainer_email='gg.gilerv@gmail.com',
    description='Mapeo (SLAM) + planificacion global Theta* con suavizado B-Spline sobre el simulador AutoDRIVE',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'global_path_publisher = global_planner.global_path_publisher:main',
            'generate_trajectory = global_planner.generate_trajectory:main',
            'smooth_trajectory = global_planner.smooth_trajectory:main',
            'generate_gif = global_planner.generate_gif:main',
        ],
    },
)
