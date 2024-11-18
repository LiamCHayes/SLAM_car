"""
Planner Function: Low Level Planner to drive robot between xstart to xgoal
"""
import numpy as np
import math

class Planner:
    """
    Planner Class:
    1.) Initialize Variables (Passed as a Column Vector X and Y)
    2.) Convert rectangular coordinates to polar coordinates (Radius and Angle)
    3.) Calibrate and convert distance to time
    4.) Controller car passes drive commands
    """
    def __init__(self, x_start, x_goal):
        """
        1.) Initialize Variables
        """
        self.x_start = x_start
        self.x_goal = x_goal

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

if __name__ == '__main__':
    x_start = np.array([[0],[0]])
    x_goal = np.array([[1],[1]])
    planner = Planner(x_start, x_goal)
    print(planner.rect_to_polar())
