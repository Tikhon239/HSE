#! /usr/bin/env python

import numpy as np

import rospy
from rospy.topics import Publisher, Subscriber
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData


class ScanVisualization:
    def __init__(self):
        self.map_size = 10
        self.resolution = 0.1

        rospy.init_node("scan_visualization")

        Subscriber("/base_scan", LaserScan, self.visualize_grid)
        self.marker_pub = Publisher("/visualization_marker", Marker, queue_size=1)
        self.map_pub = Publisher("/map", OccupancyGrid, queue_size=1)

        self.marker = self.get_marker()
        self.grid = self.get_grid()

    def get_marker(self):
        marker = Marker()

        marker.header.frame_id = "base_laser_link"

        marker.type = 8
        marker.id = 0
        marker.action = 0

        # Set the scale of the marker
        marker.scale.x = self.resolution
        marker.scale.y = self.resolution
        marker.scale.z = self.resolution

        # Set the color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0

        return marker

    def get_grid(self):
        grid = OccupancyGrid()
        grid.info = MapMetaData()

        grid.header.frame_id = "base_laser_link"
        grid.info.resolution = self.resolution

        grid.info.height = int(self.map_size / self.resolution)
        grid.info.width = int(self.map_size / self.resolution)

        grid.info.origin.position.x = -int(self.map_size / 2)
        grid.info.origin.position.y = -int(self.map_size / 2)
        grid.info.origin.position.z = 0

        return grid

    def point_distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def points_filter(self, points):
        filtered_points = []
        for prev_p, cur_p, next_p in zip(points, points[1:], points[2:]):
            prev_cur_dist = self.point_distance(prev_p, cur_p)
            cur_next_dist = self.point_distance(cur_p, next_p)
            prev_next_dist = self.point_distance(prev_p, next_p)
            if min(prev_cur_dist, cur_next_dist) / prev_next_dist < 0.75:
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

    def visualize_grid(self, msg):
        voices_map = np.zeros((self.grid.info.height, self.grid.info.width), dtype=np.int8)

        angle = msg.angle_min
        for distance in msg.ranges:
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            if (abs(x) < int(self.map_size / 2) and abs(y) < int(self.map_size / 2)):
                coord_x = int((x + int(self.map_size / 2)) / self.resolution)
                coord_y = int((y + int(self.map_size / 2)) / self.resolution)
                voices_map[coord_y, coord_x] += 1

            angle += msg.angle_increment


        self.grid.data = voices_map.ravel()
        self.map_pub.publish(self.grid)

if __name__ == "__main__":
    ScanVisualization()
    rospy.spin()
