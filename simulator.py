"""
Simulator of the environment and car. Used for NN training.
"""

###################
# Imports and Setup
###################
import numpy as np
import matplotlib.pyplot as plt
import torch


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

    def get_free_area(self):
        """
        Get indices of the free space in the map

        Returns:
            indices (numpy array): 2 x indices where first row is row index and second row is column index
        """
        indices = np.zeros((2, 1))
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                if self.is_within_walls((row, col)) and self.map[row, col] == 0:
                    indices = np.append(indices, np.array([[row], [col]]), axis=1)
        indices = indices[:, 1:]

        return indices

    def is_within_walls(self, point):
        """
        Checks if a point is within the boundries of the wall

        args:
            point (tupple): point we are testing (row, col)

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

    def pad_map(self, pad):
        """
        Pads map so lidar readings don't go out of bounds
        
        args:
            pad (int): amount to pad by
        """
        self.map = np.pad(self.map, ((pad, pad), (pad, pad)), mode='constant', constant_values=-1)

    def plot(self):
        """
        Plots the empty map
        """
        plt.imshow(self.map)
        plt.show()


class Car:
    """
    Object for a car mapping the space. The location of the car is defined by cartesian coordinates.
    The car's coordinate system is defined by the initial position of the car. The initial position
    of the car is the origin (0, 0) for its coordinate system.
    
    args:
        lidar_radius (int): radius of the lidar, positioned in the center of the car

    attributes:
        lidar_radius (int): radius of the lidar, positioned in the center of the car
        map (numpy array): The map according to the car. Only uses sensor data to create this map.
        lidar_reading (numpy array): most recent lidar reading
        path (numpy array): 2 x num_steps estimated path of the car
        coordinates (tupple): coordinates of the car in its map estimation
        np_origin_coordinates (tupple): coordinates of the map origin in the numpy array map
        prev_np_origin_coordinates (tupple): coordinates of the map origin in the numpy array map in the previous ste[]
    """
    def __init__(self, lidar_radius):
        self.lidar_radius = lidar_radius
        self.map = None
        self.lidar_reading = None
        self.path = None
        self.coordinates = None
        self.np_origin_coordinates = None
        self.prev_np_origin_coordinates = None

    def update_location(self, move_vector, add_noise):
        """
        Updates location with some optional noise to simulate real world updating

        args:
            move_vector (tupple): vector the car is moving in
            add_noise (bool): add noise to position estimation?
        """
        # Add noise
        noise_row = int(np.round(np.random.normal(0, 1))) if add_noise else 0
        noise_col = int(np.round(np.random.normal(0, 1))) if add_noise else 0
        move_row = -move_vector[0] + noise_row
        move_col = move_vector[1] + noise_col

        # Update variables
        self.coordinates = (self.coordinates[0] + move_row,
                            self.coordinates[1] + move_col)
        self.path = np.append(self.path, np.array([[self.coordinates[0]],[self.coordinates[1]]]), axis=1)

        self.prev_np_origin_coordinates = self.np_origin_coordinates
        origin_np_index_row = np.max(self.path[0]) + self.lidar_radius
        origin_np_index_col = -(np.min(self.path[1]) - self.lidar_radius)
        self.np_origin_coordinates = (origin_np_index_row, origin_np_index_col)

    def read_lidar(self, ground_truth_map: SimulatedMap, ground_truth_location):
        """
        Gets a lidar reading for the current location in the map

        args:
            ground_truth_map (SimulatedMap object): map that the car is in
            ground_truth_location (tupple): location of the car in the simulated map coordinate system (row, column)

        returns:
            lidar_reading (numpy array): value of the lidar reading at the location of the car
        """
        lidar_reading = np.full((self.lidar_radius*2+1, self.lidar_radius*2+1), -1)

        # Loop through indices and record the lidar reading
        theta_list = np.array([])
        r_list = np.array([])
        li_idx_list = []
        gt_list = []
        for li_row, row in enumerate(np.arange(ground_truth_location[0]-self.lidar_radius, ground_truth_location[0]+self.lidar_radius+1)):
            for li_col, col in enumerate(np.arange(ground_truth_location[1]-self.lidar_radius, ground_truth_location[1]+self.lidar_radius+1)):
                # Calculate theta and r
                x = li_col - self.lidar_radius
                y = self.lidar_radius - li_row
                theta = np.arctan2(y, x)
                r = (x**2 + y**2)**0.5
                
                theta_list = np.append(theta_list,theta)
                r_list = np.append(r_list,r)
                li_idx_list.append((li_row, li_col))
                gt_list.append((row, col))

        gt_list = np.array(gt_list)
        li_idx_list = np.array(li_idx_list)
        for i, li_idx in enumerate(li_idx_list):
            theta = theta_list[i]
            r = r_list[i]

            # Get ground truth locations that have theta similar to this theta and r less than this r
            epsilon = 0.1
            mask = theta-epsilon < theta_list
            theta_similar = theta_list[mask]
            matching_r = r_list[mask]
            matching_gt = gt_list[mask]
            mask2 = theta_similar < theta+epsilon
            theta_similar = theta_similar[mask2]
            matching_r = matching_r[mask2]
            matching_gt = matching_gt[mask2]
            mask3 = matching_r <= r
            matching_r = matching_r[mask3]
            matching_gt = matching_gt[mask3]

            # List of ground truth values satisfying conditions
            values = np.array([])
            for gt in matching_gt:
                values = np.append(values, ground_truth_map.map[gt[0], gt[1]])
            
            # Set this lidar reading value
            if values[values > 0].any():
                lidar_reading[li_idx[0], li_idx[1]] = -1
            else:
               lidar_reading[li_idx[0], li_idx[1]] = 0

        self.lidar_reading = lidar_reading
        return lidar_reading

    def record_lidar(self, lidar_reading):
        """
        Adds lidar reading to the map at the cars current location (according to its calcluation)

        args:
            lidar_reading (numpy array): lidar reading to append to map
        """
        if self.map is None:
            self.map = lidar_reading
            self.path = np.array([[0], [0]])
            self.coordinates = (0, 0)
            self.np_origin_coordinates = (self.lidar_radius, self.lidar_radius)
        else:
            # calculate the nececary size of the new map
            row_dim = (np.max(self.path[0]) + self.lidar_radius) - (np.min(self.path[0]) - self.lidar_radius)
            col_dim = (np.max(self.path[1]) + self.lidar_radius) - (np.min(self.path[1]) - self.lidar_radius)

            # calculate where we are currently in relation to the numpy array by
            # finding where we are in the relation to the origin
            coord_np_index_row = self.np_origin_coordinates[0] - self.coordinates[0]
            coord_np_index_col = self.np_origin_coordinates[1] + self.coordinates[1]

            # Map the old map (previous step) onto the new map (current step)
            new_map = np.full((row_dim+1, col_dim+1), -1)
            coord_shift_row = self.np_origin_coordinates[0] - self.prev_np_origin_coordinates[0]
            coord_shift_col = self.np_origin_coordinates[1] - self.prev_np_origin_coordinates[1]
            for row in range(self.map.shape[0]):
                for col in range(self.map.shape[1]):
                    new_map[row + coord_shift_row, col + coord_shift_col] = self.map[row, col]

            # Add the lidar reading to the new map
            new_map_reading_idx_row = np.arange(coord_np_index_row - self.lidar_radius, coord_np_index_row + self.lidar_radius + 1)
            new_map_reading_idx_col = np.arange(coord_np_index_col - self.lidar_radius, coord_np_index_col + self.lidar_radius + 1)
            for lidar_idx_row, row in enumerate(new_map_reading_idx_row):
                for lidar_idx_col, col in enumerate(new_map_reading_idx_col):
                    if new_map[row, col] == -1:
                        new_map[row, col] = lidar_reading[lidar_idx_row, lidar_idx_col]
                    else:
                        if lidar_reading[lidar_idx_row, lidar_idx_col] != -1:
                            new_map[row, col] = new_map[row, col] + lidar_reading[lidar_idx_row, lidar_idx_col]
            self.map = new_map

    def plot(self):
        """
        Plots the current car knowledge
        """
        plt.imshow(self.map)
        plt.title('Current Map - Knowledge of Car')
        plt.show()


class Simulator:
    """
    Object to run interactions between car and environment.

    args:
        simulated_map (SimulatedMap object): ground truth map object
        car (Car object): object of car in map

    attributes:
        simulated_map (SimulatedMap object): ground truth map object
        car (Car object): object of car in map
        ground_truth_location (tupple): Location of the car in the ground truth map (row, column)
        path (numpy array): 2 x num_steps ground truth path of the car
    """
    def __init__(self, simulated_map: SimulatedMap):
        self.simulated_map = simulated_map
        self.car = None
        self.ground_truth_location = None
        self.path = None

    def get_car_map(self):
        """
        Returns car map (self.car.map)
        """
        return self.car.map
    
    def get_ground_truth(self):
        """
        Returns ground truth map (self.simulated_map)
        """
        return self.simulated_map
    
    def get_global_coordinates(self):
        """
        Returns global coordinates (self.ground_truth_location)
        """
        return self.ground_truth_location
    
    def get_path(self):
        """
        Returns path through environment (self.path)
        """
        return self.path

    def spawn_car(self, lidar_radius, plot=False):
        """
        Spawns the car in the map in a random spot

        args:
            lidar_radius (int): radius of the lidar, positioned in the center of the car
            plot (bool): True if you want to see a map with the car mapped on it when spawned
        
        returns:
            car (Car object): car spawned in the environment
            spawn_point (tupple): spawn location in the ground truth map (row, col)
        """
        # Pad map and get free area and pick a random sample from this area
        self.simulated_map.pad_map(lidar_radius)
        free_area = self.simulated_map.get_free_area()

        # Spawn until there is no collision
        in_free_space = False
        while not in_free_space:
            idx = np.random.randint(0, free_area.shape[1])
            spawn_point = free_area[:, idx]
            car_radius = 5
            free_area_row = free_area[1, np.where(free_area[0, :] == spawn_point[0])[0]]
            free_area_col = free_area[0, np.where(free_area[1, :] == spawn_point[1])[0]]

            temp_flag = True
            for val in np.arange(spawn_point[1] - car_radius, spawn_point[1] + car_radius):
                if val not in free_area_row:
                    temp_flag = False
            for val in np.arange(spawn_point[0] - car_radius, spawn_point[0] + car_radius):
                if val not in free_area_col:
                    temp_flag = False
            in_free_space = temp_flag
        spawn_point = (int(spawn_point[0]), int(spawn_point[1]))

        # initialize attributes
        self.car = Car(lidar_radius)
        self.ground_truth_location = spawn_point
        self.car.record_lidar(self.car.read_lidar(self.simulated_map, spawn_point))
        self.path = np.array([[spawn_point[0]], [spawn_point[1]]])

        # Plot car if desired
        if plot:
            self.plot_car()
            self.car.plot()

    def step(self, move_vector, add_noise, plot=False):
        """
        Do a single time step
            1. Execute movement
            2. Read from LIDAR
            3. Record LIDAR measurement in the car map

        args:
            move_vector (tupple): Where to go next (change in row, change in col) in the car reference frame
            add_noise (bool): Add noise to car state estimation if desired
            plot (bool): True if you want to see a map with the car mapped on it when spawned

        returns:
            success (bool): True if step is successful, False otherwise
            lidar_reading (numpy array): If step is not successful, returns array of all -1
        """
        # 1. execute movement
        if isinstance(move_vector, torch.Tensor):
            move_vector = move_vector.cpu()
            move_vector = move_vector.detach().numpy()[0]
            move_vector = move_vector.astype(np.int32)
        try:
            move_vector = np.round(move_vector).astype(np.int32)
            self.ground_truth_location = (self.ground_truth_location[0] + move_vector[0],
                                          self.ground_truth_location[1] + move_vector[1])
        except:
            print("\nInvalid move vector input. Skipping this step.")
            print("Move vector: ", move_vector)
            return True, np.full((self.car.lidar_radius*2+1, self.car.lidar_radius*2+1), -1)

        self.path = np.append(self.path, np.array([[self.ground_truth_location[0]], [self.ground_truth_location[1]]]), axis=1)
        if self.check_collision():
            # print("COLLISION: Car has collided with a wall or obstacle")
            return False, np.full((self.car.lidar_radius*2+1, self.car.lidar_radius*2+1), -1)
        self.car.update_location(move_vector, add_noise)
        
        # 2. Read from LIDAR
        lidar_reading = self.car.read_lidar(self.simulated_map, self.ground_truth_location)

        # 3. Record LIDAR measurement in car map
        self.car.record_lidar(lidar_reading)

        # Plot if desired
        if plot:
            self.plot_path()
            self.car.plot()

        return True, lidar_reading

    def check_collision(self):
        """
        Checks if the car has collided with an obstacle or wall
        
        Returns:
            flag (bool): True if car is in a forbidden area, False otherwise
        """
        flag = False
        try:
            map_val = self.simulated_map.map[self.ground_truth_location[0], self.ground_truth_location[1]]
            if map_val != 0:
                flag = True
        except:
            flag = True        

        return flag

    def plot_car(self):
        """
        Plots the ground truth car location
        """
        new_map = np.copy(self.simulated_map.map)
        for row in np.arange(self.ground_truth_location[0]-2, self.ground_truth_location[0]+2):
            for col in np.arange(self.ground_truth_location[1]-2, self.ground_truth_location[1]+2):
                new_map[row, col] = 2
        plt.imshow(new_map)
        plt.title("Ground Truth Car Position")
        plt.show()

    def plot_path(self):
        """
        Plots the ground truth path of the car through the environment
        """
        new_map = np.copy(self.simulated_map.map)
        for path_idx in range(self.path.shape[1]):
            path = self.path[:, path_idx]
            for row in np.arange(path[0]-2, path[0]+2):
                for col in np.arange(path[1]-2, path[1]+2):
                    new_map[row, col] = 2
        plt.imshow(new_map)
        plt.title("Ground Truth Car Path")
        plt.show()