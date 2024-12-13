#!/usr/bin/env python3
import rospy
import numpy as np
import math
import time

import os
# os.chdir('../..')
from planner.srv import *
from planner.srv import complete_srv, complete_srvResponse
def handle_complete(req):
    
    return complete_srvResponse('complete')

if __name__ == "__main__":
    unit_distance_movement = 1.3667 # time to travel 1 meter t/m
    unit_angle_change_rad = 0.0995 # time to rotate 2pi rad t/rad
    unit_ange_change_deg = 0.001736 # time to rotate 360 deg t/deg
    print('waiting')
    while True:
        rospy.wait_for_service('NN_goal')
        print('serviced')
        state_goal_req = rospy.ServiceProxy('NN_goal', coords_srv)
        state_goal_resp = state_goal_req(0,0)
        state_goal = np.array([[state_goal_resp.x_goal/100],[state_goal_resp.y_goal/100]])
        
        print(state_goal)
        time.sleep(5)
