#! /usr/bin/env python

import numpy as np

import rospy
from rospy.topics import Publisher, Subscriber
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan


class ScanVisualization:
    def __init__(self):
        rospy.init_node("scan_visualization")

        Subscriber("/base_scan", LaserScan, self.visualize_points)
        self.marker_pub = Publisher("/visualization_marker", Marker, queue_size = 1)

        self.marker = self.get_marker()

    def get_marker(self):
        marker = Marker()

        marker.header.frame_id = "base_laser_link"

        marker.type = 8
        marker.id = 0
        marker.action = 0

        # Set the scale of the marker
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Set the color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0

        return marker

    def point_distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def points_filter(self, points):
        filtered_points = []
        for prev_p, cur_p, next_p in zip(points, points[1:], points[2:]):
            prev_cur_dist = self.point_distance(prev_p, cur_p)
            cur_next_dist = self.point_distance(cur_p, next_p)
            prev_next_dist = self.point_distance(prev_p, next_p)
            if min(prev_cur_dist, cur_next_dist) < 2 * prev_next_dist:
                filtered_points.append(cur_p)
        return filtered_points

    def visualize_points(self, msg):
        points = []
        angle = msg.angle_min
        for distance in msg.ranges:
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            points.append(Point(x, y, 0))

            angle += msg.angle_increment

        self.marker.points = self.points_filter(points)
        self.marker_pub.publish(self.marker)

if __name__ == "__main__":
    ScanVisualization()
    rospy.spin()
