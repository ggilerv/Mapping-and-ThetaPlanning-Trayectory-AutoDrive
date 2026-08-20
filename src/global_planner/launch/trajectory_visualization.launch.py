"""Brings up everything needed to visualize the Theta* global trajectory
over the SLAM map, together with the AutoDRIVE simulator:

  - nav2_map_server (+ lifecycle manager) serving maps/F1tenth_Map.yaml on
    /map, so RViz can show the actual SLAM-built occupancy grid.
  - The AutoDRIVE ROS 2 bridge nodes (incoming/outgoing), same as
    autodrive_f1tenth's own simulator_bringup_rviz.launch.py, so this can
    run against the live AutoDRIVE Simulator app.
  - global_path_publisher, publishing the pre-computed Theta* trajectory
    (waypoints/theta_star_waypoints_smooth.csv) on /global_path +
    /global_path_markers.
  - RViz, with a config that shows the map, the LiDAR scan and the path/marker.

Start the AutoDRIVE Simulator app yourself first, then run this launch file.
"""
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    share = Path(get_package_share_directory('global_planner'))
    map_yaml = str(share / 'maps' / 'F1tenth_Map.yaml')
    waypoints_csv = str(share / 'waypoints' / 'theta_star_waypoints_smooth.csv')

    return LaunchDescription([
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_yaml, 'frame_id': 'map'}],
        ),
        # map_server needs a moment to finish creating its lifecycle service
        # interfaces before the manager can configure it -- starting both at
        # once races the manager ahead of the node it's managing and the
        # bringup fails ("Failed to change state for node: map_server").
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='nav2_lifecycle_manager',
                    executable='lifecycle_manager',
                    name='lifecycle_manager_map_server',
                    output='screen',
                    parameters=[{'autostart': True, 'node_names': ['map_server']}],
                ),
            ],
        ),
        Node(
            package='autodrive_f1tenth',
            executable='autodrive_incoming_bridge',
            name='autodrive_incoming_bridge',
            emulate_tty=True,
            output='screen',
        ),
        Node(
            package='autodrive_f1tenth',
            executable='autodrive_outgoing_bridge',
            name='autodrive_outgoing_bridge',
            emulate_tty=True,
            output='screen',
        ),
        Node(
            package='global_planner',
            executable='global_path_publisher',
            name='global_path_publisher',
            output='screen',
            parameters=[{'waypoints_file': waypoints_csv, 'frame_id': 'map'}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            arguments=['-d', [FindPackageShare('global_planner'), '/rviz', '/trajectory.rviz']],
        ),
    ])
