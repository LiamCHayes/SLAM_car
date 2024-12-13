#!/usr/bin/env python3
import rospy
import os
import numpy as np
import math
import time
#os.chdir()
from planner.srv import coords_srv, coords_srvResponse
from planner.srv import *
def handle_coords(req):
    print(req)
    return coords_srvResponse(action[0], action[1])

if __name__ == "__main__":
    rospy.init_node('NN_goal')
    rospy.Service('NN_goal', coords_srv, handle_coords)
    while True:
        global action
        action = (100, 50)
