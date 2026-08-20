#!/usr/bin/env python3
"""Publishes the pre-computed global trajectory (Theta*, over the SLAM map)
as a nav_msgs/Path, for visualization in RViz alongside the AutoDRIVE
simulator.

Loads an (x, y) waypoint CSV -- generated offline by generate_trajectory.py
/ smooth_trajectory.py -- and republishes it periodically on /global_path
(nav_msgs/Path) and /global_path_markers (visualization_msgs/MarkerArray:
the LINE_STRIP plus a green START sphere at waypoints[0] and a red END
sphere at waypoints[-1], so which end is which is visible in RViz instead
of having to guess from the CSV row order) in the `map` frame, the same
frame the SLAM-built occupancy map is served in.
"""
import csv
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray


def load_waypoints(csv_path):
    waypoints = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            x, y = float(row[0]), float(row[1])
            waypoints.append((x, y))
    return waypoints


class GlobalPathPublisher(Node):

    def __init__(self):
        super().__init__('global_path_publisher')

        default_csv = str(Path(get_package_share_directory('global_planner'))
                           / 'waypoints' / 'theta_star_waypoints_smooth.csv')
        self.declare_parameter('waypoints_file', default_csv)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_rate', 1.0)

        csv_path = self.get_parameter('waypoints_file').value
        self.frame_id = self.get_parameter('frame_id').value
        rate = self.get_parameter('publish_rate').value

        self.waypoints = load_waypoints(csv_path)
        self.get_logger().info(
            f'Loaded {len(self.waypoints)} waypoints from {csv_path}')

        latched_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(PathMsg, '/global_path', latched_qos)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/global_path_markers', latched_qos)

        self.path_msg = self._build_path_msg()
        self.marker_msg = self._build_marker_array()

        # Republish on a timer too, in case a subscriber's QoS doesn't match
        # transient local durability (e.g. RViz's default Displays panel).
        self.timer = self.create_timer(1.0 / rate, self.publish_path)
        self.publish_path()

    def _build_path_msg(self):
        path = PathMsg()
        path.header.frame_id = self.frame_id
        for x, y in self.waypoints:
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path

    def _build_marker_array(self):
        line = Marker()
        line.header.frame_id = self.frame_id
        line.ns = 'global_path'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.04
        line.color.r = 0.1
        line.color.g = 0.9
        line.color.b = 0.1
        line.color.a = 1.0
        for x, y in self.waypoints:
            p = PoseStamped().pose.position
            p.x, p.y, p.z = x, y, 0.02
            line.points.append(p)

        def endpoint_marker(marker_id, x, y, r, g, b):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.ns = 'global_path'
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            return m

        x0, y0 = self.waypoints[0]
        x1, y1 = self.waypoints[-1]
        start_marker = endpoint_marker(1, x0, y0, 0.0, 1.0, 0.0)  # green = start
        end_marker = endpoint_marker(2, x1, y1, 1.0, 0.0, 0.0)  # red = end

        return MarkerArray(markers=[line, start_marker, end_marker])

    def publish_path(self):
        now = self.get_clock().now().to_msg()
        self.path_msg.header.stamp = now
        for m in self.marker_msg.markers:
            m.header.stamp = now
        self.path_pub.publish(self.path_msg)
        self.marker_pub.publish(self.marker_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
