#! /usr/bin/python3

import rospy
from geometry_msgs.msg import Twist

rospy.init_node('turtle_square')

pub = rospy.Publisher('/turtle1/cmd_vel',
    Twist,
    queue_size=1
)

msg = Twist()

r = rospy.Rate(0.5) #Hz
while not rospy.is_shutdown():
    msg.linear.x = 1.0
    msg.angular.z = 0.0

    pub.publish(msg)
    r.sleep()

    msg.linear.x = 0.0
    msg.angular.z = 1.57

    pub.publish(msg)
    r.sleep()

