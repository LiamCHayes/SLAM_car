#!/usr/bin/env python3

import rospy
import numpy as np
import math
import time
from gpiozero import Motor
from planner.srv import *

def distance_calc(state_goal):
    x_goal = state_goal[0,0]
    y_goal = state_goal[1,0]

    #if NN local
    dist = np.sqrt((x_goal)**2 + (y_goal)**2)
    if x_goal != 0:
        if x_goal > 0 and y_goal >= 0:
            angle_desired = math.atan(y_goal/x_goal) 
        elif x_goal < 0:
            angle_desired = math.atan(y_goal/x_goal) + np.pi
        elif x_goal > 0 and y_goal < 0:
            angle_desired = math.atan(y_goal/x_goal) + np.pi * 2
    else:
        if y_goal > 0:
            angle_desired = np.pi/2
        elif y_goal < 0:
            angle_desired = 3*np.pi/2
    #print(angle_desired)

    return dist, angle_desired


def serial_write(dist, angle_desired, dist_unit, angle_unit):
    left_side_front = Motor(forward=27, backward=14) ## TEMP
    left_side_back = Motor(forward=17,backward=23) ## TEMP
    right_side_front = Motor(forward=19, backward=16) ## TEMP
    right_side_back = Motor(forward=26,backward=20) ## TEMP

    dist_time = round(dist * dist_unit, 2)
    angle_time = round(angle_desired * angle_unit, 2)

    #Angle serial commands
    timer = angle_time
    left_side_back.stop()
    left_side_front.stop()
    right_side_back.stop()
    right_side_front.stop()
    time.sleep(2)
    print(angle_desired)
    if angle_desired >= np.pi:
        left_side_front.backward()
        left_side_back.backward()
        right_side_front.forward()
        right_side_back.forward()
    else:
        left_side_front.forward()
        left_side_back.forward()
        right_side_front.backward()
        right_side_back.backward()
    time.sleep(timer)
    timer = 0
    left_side_front.stop()
    left_side_back.stop()
    right_side_front.stop()
    right_side_back.stop()
    
    #Dist serial commands
    time.sleep(2)
    timer = dist_time
    left_side_front.forward()
    left_side_back.forward()
    right_side_front.forward()
    right_side_back.forward()
    time.sleep(timer)
    timer = 0
    left_side_front.stop()
    right_side_front.stop()
    left_side_back.stop()
    right_side_back.stop()



if __name__ == "__main__":
    unit_distance_movement = 1.3667 # time to travel 1 meter t/m
    unit_angle_change_rad = 0.0995 # time to rotate 2pi rad t/rad
    unit_angle_change_deg = 0.001736 # time to rotate 360 deg t/deg

    print('waiting')
    while True:
        rospy.wait_for_service('NN_goal', timeout=30)
        print('serviced')
        state_goal_req = rospy.ServiceProxy('NN_goal', coords_srv)
        state_goal_resp = state_goal_req(0,0)
        state_goal = np.array([[state_goal_resp.x_goal/100],[state_goal_resp.y_goal/100]])
        print(state_goal)
        
        dist_desired, angle_desired = distance_calc(state_goal)
        serial_write(dist_desired, angle_desired, unit_distance_movement, unit_angle_change_rad)
        time.sleep(3)
