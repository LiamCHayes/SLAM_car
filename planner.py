"""
Planner Function: Low Level Planner to drive robot between xstart to xgoal
"""
import math
import numpy as np
from gpiozero import Motor
from time import sleep

class Planner:
    """
    Planner Class:
    1.) Initialize Variables (Passed as a Column Vector X and Y)
    2.) Convert rectangular coordinates to polar coordinates (Radius and Angle)
    3.) Calibrate and convert distance to time
    4.) Controller car passes drive commands
    """
    def __init__(self, xstart, xgoal):
        """
        1.) Initialize Variables
        """
        self.x_start = xstart
        self.x_goal = xgoal

    def rect_to_polar(self):
        """
        2.) Convert Rectangular Coordinate to Polar Coordinates
        """
        rect_coords = np.array([[(self.x_goal[0] - self.x_start[0])],
                                [self.x_goal[1] - self.x_start[1]]])
        radius = math.sqrt(((rect_coords[0])**2)+((rect_coords[1])**2))
        #In radians
        theta = math.atan((abs(rect_coords[1]))/(abs(rect_coords[0])))
        polar_coords = np.array([[radius],[theta]])

        return polar_coords

    def dist_to_time(self):
        """
        Lower level planner to convert distance and angle into time.
        This sends the robot from its current position to goal position.
        """
        polar_coords = self.rect_to_polar()
        # distance/time constant calculated by calibration
        # eventually automatic protocol currently manual
        time_constant = 1
        radius_time = polar_coords[0] * time_constant
        angle_time = polar_coords[1] * time_constant
        time_coords = np.array([[radius_time],[angle_time]])

        return time_coords

    def controller(self):
        """
        Controller Function (Looking at voltage reduction to prevent overshooting)
        """
        h_bridge_1 = Motor(17,18)
        h_bridge_2 = Motor(20,21)

        time_duration = self.dist_to_time()
        #Rotation
        h_bridge_1.backward()
        h_bridge_2.forward()
        sleep(time_duration[1])

        h_bridge_1.stop()
        h_bridge_2.stop()
        sleep(1)

        #Forward
        h_bridge_1.forward()
        h_bridge_2.forward()
        sleep(time_duration[0])

        h_bridge_1.stop()
        h_bridge_2.stop()
        sleep(1)
