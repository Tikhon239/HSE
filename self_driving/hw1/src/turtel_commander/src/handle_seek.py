#! /usr/bin/python3

import numpy as np

import rospy
from turtlesim.msg import Pose
from turtlesim.srv import Spawn
from geometry_msgs.msg import Twist


class HandleSeek:
    def __init__(self):
        self.hunter_pos = np.zeros(2)
        self.hunter_view = np.array([1.0, 0.0])

        rospy.Subscriber("/turtle1/pose", Pose, self.move_to)
        rospy.Subscriber("/turtle2/pose", Pose, self.update_pos)

        self.pub = rospy.Publisher('turtle2/cmd_vel', Twist, queue_size=1)
        #self.r = rospy.Rate(0.5) #Hz

    def update_pos(self, msg):
        self.hunter_pos = np.array([
            msg.x,
            msg.y
        ])
        self.hunter_view = np.array([
            np.cos(msg.theta),
            np.sin(msg.theta)
        ])

    def move_to(self, msg):
        victim_pos = np.array([
            msg.x,
            msg.y
        ])

        diff = victim_pos - self.hunter_pos
        diff_norm = np.linalg.norm(diff)

        res_msg = Twist()
        res_msg.linear.x = 0.0
        res_msg.angular.z = 0.0

        if diff_norm > 0.1:
            res_msg.linear.x = diff_norm
            diff /= diff_norm
            angle = np.arccos(diff.dot(self.hunter_view))
            direction = np.sign(np.cross(self.hunter_view, diff))
            res_msg.angular.z = direction * angle

        self.pub.publish(res_msg)
        # это все почему-то ломает
        #self.r.sleep()


if __name__ == "__main__":    
    rospy.init_node("handle_seek")
    
    rospy.wait_for_service('/spawn')
    spawn_func = rospy.ServiceProxy('/spawn',Spawn)
    spawn_func(4.0, 4.0, 0.0, 'turtle2')

    HandleSeek()

    rospy.spin()