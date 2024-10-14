"""
Classes to generate ground truth data for NN simulation training
"""

###################
# Imports and Setup
###################
import numpy as np


#########
# Classes
#########
class SimulatedMap:
    """
    Object for a simulated ground truth map. Origin is in top, left-hand corner of the map i.e. row 0, column 0

    args:
        size (tupple): a tupple of length 2 that represents the size of the generated map. (height, width)

    attributes:
        size (tupple): (height, width)
        map (numpy array): numpy array representing map. -1 is unreachable space, 0 is free space, 1 is obstacle / wall
    """
    def __init__(self, size):
        self.size = size
        self.map = None

    def get_vertices(self):
        """
        Gets vertices of the available area

        Returns:
             v_1, v_2, v_3, v_4: Tupples with coordinates of each vertex, going clockwise starting from top left quadrant
        """
        # Top left quadrant vertex
        y = int(np.round(np.random.triangular(0, 1, self.size[0]/4)))
        x = int(np.round(np.random.triangular(0, 1, self.size[1]/4)))
        v_1 = (y, x)

        # Top right quadrant vertex
        y = int(np.round(np.random.triangular(0, 1, self.size[0]/4)))
        x = int(np.round(np.random.triangular(3*self.size[1]/4, self.size[1]-2, self.size[1]-1)))
        v_2 = (y, x)

        # Bottom right quadrant vertex
        y = int(np.round(np.random.triangular(3*self.size[0]/4, self.size[0]-2, self.size[0]-1)))
        x = int(np.round(np.random.triangular(3*self.size[1]/4, self.size[1]-2, self.size[1]-1)))
        v_3 = (y, x)

        # Bottom left quadrant vertex
        y = int(np.round(np.random.triangular(3*self.size[0]/4, self.size[0]-2, self.size[0]-1)))
        x = int(np.round(np.random.triangular(0, 1, self.size[1]/4)))
        v_4 = (y, x)

        return v_1, v_2, v_3, v_4

    def get_walls(self, v1, v2, v3, v4):
        """
        Returns the coordinates of the walls (boundaries) of the current map

        args:
            v1, v2, v3, v4: Tupples of vertices starting in the top left and going around clockwise

        Returns:
            coordinates: Numpy array (2 x n) where n is the number of coodinates. First row is row index, second is col index.
        """
        # Top edge
        rows = np.round(np.linspace(v1[0], v2[0], abs(v1[1]-v2[1])+1)) # Row values
        cols = np.round(np.linspace(v1[1], v2[1], abs(v1[1]-v2[1])+1)) # Column values

        # Right edge
        rows = np.concatenate((rows, np.round(np.linspace(v2[0], v3[0], abs(v2[0]-v3[0])+1)))) # Row values
        cols = np.concatenate((cols, np.round(np.linspace(v2[1], v3[1], abs(v2[0]-v3[0])+1)))) # Column values

        # Bottom edge
        rows = np.concatenate((rows, np.round(np.linspace(v3[0], v4[0], abs(v3[1]-v4[1])+1)))) # Row values
        cols = np.concatenate((cols, np.round(np.linspace(v3[1], v4[1], abs(v3[1]-v4[1])+1)))) # Column values

        # Left edge
        rows = np.concatenate((rows, np.round(np.linspace(v4[0], v1[0], abs(v4[0]-v1[0])+1)))) # Row values
        cols = np.concatenate((cols, np.round(np.linspace(v4[1], v1[1], abs(v4[0]-v1[0])+1)))) # Column values

        rows = [int(rows[i]) for i in range(len(rows))]
        cols = [int(cols[i]) for i in range(len(cols))]

        coordinates = np.vstack((rows, cols))

        return coordinates

    def is_within_walls(self, point):
        """
        Checks if a point is within the boundries of the wall

        Returns:
            flag (bool): True if the point is within the walls
        """
        row = point[0]
        col = point[1]
        flag = False

        # Get wall locations for this row
        wall_locations_row = np.where(self.map[row, :] == 1)[0]
        if len(wall_locations_row) < 2:
            return flag
        max_row_wall = np.max(wall_locations_row)
        min_row_wall = np.min(wall_locations_row)

        # Get wall locations for this column
        wall_locations_column = np.where(self.map[:, col] == 1)[0]
        if len(wall_locations_column) < 2:
            return flag
        max_column_wall = np.max(wall_locations_column)
        min_column_wall = np.min(wall_locations_column)

        if (min_column_wall < row < max_column_wall) and (min_row_wall < col < max_row_wall):
            flag = True
        
        return flag

    def create_map(self):
        """
        Creates a random 4-wall map with no obstacles. 

        Returns:
            ground_truth (numpy array): Represents map with 0 as a free space, 1 as an obstacle, and -1 as an unreachable space
        """
        if self.map is not None:
            return self.map

        # Base map full of -1
        ground_truth = np.full(self.size, -1)

        # Get 4 vertices
        v1, v2, v3, v4 = self.get_vertices()

        # Get coordinates of edges and plot on map
        wall_coordinates = self.get_walls(v1, v2, v3, v4)
        ground_truth[wall_coordinates[0], wall_coordinates[1]] = 1
        self.map = ground_truth

        # Set points inside walls to 0 (i.e. traversable area)
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                if self.is_within_walls((row, col)):
                    ground_truth[row, col] = 0 if ground_truth[row, col] == -1 else 1

        self.map = ground_truth
        return ground_truth

    def create_obstacles(self, n):
        """
        Generates n obstacles based on randomly generated obstacle parameters

        args:
            n (int): Number of obstacles to generate
        """
        row = np.random.randint(1, self.size[0]-1, n)
        col = np.random.randint(1, self.size[1]-1, n)
        height = np.random.randint(1, self.size[0]/4, n)
        width = np.random.randint(1, self.size[1]/4, n)

        for i in range(n):
            self.map_obstacle((row[i], col[i]), height[i], width[i])

    def map_obstacle(self, location, height, width, shape='rectangle'):
        """
        Adds obstacles to the map from the self.obstacles list

        Inputs:
            location (tupple): coordinates to place the center of the obstacle. (row, column)
            height (int): height of obstacle
            width (int): width of obstacles
            shape (string): shape of obstacle. Supported shapes ['rectangle']

        Returns:
            ground_truth (numpy array): Represents map with 0 as a free space, 1 as an obstacle, and -1 as an unreachable space
        """
        if shape == 'rectangle':
            left_boundary = location[1] - int(round(width/2))
            right_boundary = location[1] + int(round(width/2))
            top_boundary = location[0] + int(round(height/2))
            bottom_boundary = location[0] - int(round(height/2))

            for col in range(left_boundary, right_boundary+1):
                for row in range(bottom_boundary, top_boundary+1):
                    if col > self.size[1]-1 or row > self.size[0]-1 or col < 0 or row < 0:
                        continue
                    if self.is_within_walls((row, col)):
                        self.map[row, col] = 1
