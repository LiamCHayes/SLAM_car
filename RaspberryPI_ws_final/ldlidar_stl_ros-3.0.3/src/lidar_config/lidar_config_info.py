#!/usr/bin/env python3
import math
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
import numpy as np
import time

class LidarDataConfig:
    def __init__(self):
        rospy.init_node('lidar_config')
        rospy.Subscriber("/scan", LaserScan, self.callback)
        self.cart_pub = rospy.Publisher("/cartesian_coords", Pose, queue_size=10)
        self.rate = rospy.Rate(1)
        rospy.spin()
       

    def callback(self, data):
        self.data = data
        ang_inc = data.angle_increment
        ang_min = data.angle_min
        samples = len(data.ranges)
        print(samples)
        ranges = data.ranges
        rospy.loginfo(self.data)
        cartesian_x, cartesian_y, cartesian_theta = self.polar2cartesian(ranges, ang_inc, ang_min, samples)
        for i in range(0, len(cartesian_x)):
            pub_msg = Pose()
            pub_msg.position.x = cartesian_x[i]
            pub_msg.position.y = cartesian_y[i]
            pub_msg.position.z = cartesian_theta[i]
            pub_msg.orientation.x = 0
            pub_msg.orientation.y = 0
            pub_msg.orientation.z = 0
            pub_msg.orientation.w = 0
            self.cart_pub.publish(pub_msg)

        
    def polar2cartesian(self, ranges, angle_inc, angle_init, samples):
        x = []
        y = []
        theta = []
        for i in range(0, samples):
            x.append([])
            y.append([])
            theta.append([])
            angle_curr = angle_init + i * angle_inc
            theta[i] = angle_curr
            radius = ranges[i]
            x[i] = radius * math.cos(angle_curr)
            y[i] = radius * math.sin(angle_curr)
            
        return x,y,theta


if __name__ == '__main__':
    LidarDataConfig()
    
