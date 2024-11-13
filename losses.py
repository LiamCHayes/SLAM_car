"""
Losses for the networks
"""

###################
# Imports and Setup
###################
import numpy as np

import simulator


#######################
# Classes and functions
#######################
class Reward:
    """
    Functions to calculate reward for the RL network. 

    The reward is:
        + empty_reward for each new unknown pixel discovered
        + obstacle_reward for each new obstacle pixel discovered
        0 if the robot collides with an obstacle

    args:
        empty_reward (int): amount of reward for discovering an empty space
        obstacle_reward (int): amount of reward for discovering an obstacle space
        negative_reinforcement (int): amount of reward for rediscovering mapped area (negative or 0)

    attributes:
        simulator (Simulator object): simulator that we are in right now
        empty_reward (int): amount of reward for discovering an empty space
        obstacle_reward (int): amount of reward for discovering an obstacle space
        negative_reinforcement (int): amount of reward for rediscovering mapped area (negative or 0)
        reward_map (numpy array): map of available rewards
        prev_pct_explored (float): Percent of the map explored before we did the action
    """
    def __init__(self, empty_reward, obstacle_reward, negative_reinforcement):
        self.simulator = None
        self.empty_reward = empty_reward
        self.obstacle_reward = obstacle_reward
        self.negative_reinforcement = negative_reinforcement
        self.reward_map = None
        self.prev_pct_explored = 0

    def discover_reward(self, sim: simulator.Simulator):
        """
        Computes the map of rewards for the whole map. The car collects 
        the reward by reading the space with its LIDAR. Saves the map in 
        self.reward_map

        Given a the area mapped out by the robot and the ground truth map of
        the environment, compute the map of the rewards obtainable by the robot.

        args:
            sim (Simulator object): simulator that we are in right now
        """
        self.update_simulator(sim)

        # Variables we need to find the unmapped area
        car_map = self.simulator.car.map
        car_map_origin = self.simulator.car.np_origin_coordinates
        ground_truth_start = self.simulator.path[:, 0]

        car_map_shape = car_map.shape
        reward_map_shape = self.simulator.simulated_map.map.shape

        def is_in_map(row, col):
            """Returns true if is in car map"""
            row_more = row > ground_truth_start[0] - car_map_origin[0]
            row_less = row < ground_truth_start[0] + car_map.shape[0]-car_map_origin[0]
            row_inrange = row_less and row_more

            col_more = col > ground_truth_start[1] - car_map_origin[1]
            col_less = col < ground_truth_start[1] + car_map.shape[1]-car_map_origin[1]
            col_inrange = col_more and col_less

            flag = row_inrange and col_inrange

            return flag
        
        def car_coordinates(row, col):
            """Returns coordinates in the car map"""
            car_row = row - (ground_truth_start[0] - car_map_origin[0])
            car_col = col - (ground_truth_start[1] - car_map_origin[1])

            return car_row, car_col

        reward_map = np.zeros(reward_map_shape)

        # Assign rewards to unmapped area
        ground_truth = self.simulator.simulated_map.map
        for row in range(reward_map_shape[0]):
            for col in range(reward_map_shape[1]):
                if is_in_map(row, col):
                    car_row, car_col = car_coordinates(row, col)
                    reward_map[row, col] = self.empty_reward if car_map[car_row, car_col] == -1 else self.negative_reinforcement
                elif ground_truth[row, col] == 0:
                    reward_map[row, col] = self.empty_reward
                elif ground_truth[row, col] == 1:
                    reward_map[row, col] = self.obstacle_reward

        self.reward_map = reward_map

    def update_simulator(self, sim: simulator.Simulator):
        """
        Updates simulator

        args:
            sim (Simulator object): simulator that we are in right now
        """
        self.simulator = sim

    def collect_reward(self, done, sim: simulator.Simulator):
        """
        Collects reward from the map
        
        args:
            done (bool): True if it is the end of an episode
            sim (Simulator object): simulator that we are in right now
        """
        self.update_simulator(sim)

        if done:
            return -20000

        global_coords = self.simulator.get_global_coordinates()
        lidar_radius = self.simulator.car.lidar_radius
        
        reward = 0
        # Loop through indices and record ones that are within radius
        for row in np.arange(global_coords[0]-lidar_radius, global_coords[0]+lidar_radius+1):
            for col in np.arange(global_coords[1]-lidar_radius, global_coords[1]+lidar_radius+1):
                distance = ((row-global_coords[0])**2 + (col-global_coords[1])**2)**0.5
                if distance <= lidar_radius:
                    reward += self.reward_map[row, col]

        return reward

    def pct_explored(self, done, sim: simulator.Simulator):
        """
        Reward is a function of the percent of map explored
        """
        self.update_simulator(sim)

        if done:
            self.prev_pct_explored = 0
            return -100
        
        current_pct = (sim.car.map >= 0).sum() / (sim.simulated_map.map >= 0).sum() * 100
        prev_pct = self.prev_pct_explored
        pct_reward = np.ceil(current_pct) - np.floor(prev_pct)

        self.prev_pct_explored = current_pct

        return pct_reward

    def discover_sparse(self, sim: simulator.Simulator, probability):
        """
        Computes the map of rewards for the whole map. The car collects 
        the reward by reading the space with its LIDAR. Saves the map in 
        self.reward_map

        Given a the area mapped out by the robot and the ground truth map of
        the environment, compute the map of the rewards obtainable by the robot.

        This is the sparse version

        args:
            sim (Simulator object): simulator that we are in right now
        """
        self.update_simulator(sim)

        # Variables we need to find the unmapped area
        car_map = self.simulator.car.map
        car_map_origin = self.simulator.car.np_origin_coordinates
        ground_truth_start = self.simulator.path[:, 0]

        car_map_shape = car_map.shape
        reward_map_shape = self.simulator.simulated_map.map.shape

        def is_in_map(row, col):
            """Returns true if is in car map"""
            row_more = row > ground_truth_start[0] - car_map_origin[0]
            row_less = row < ground_truth_start[0] + car_map.shape[0]-car_map_origin[0]
            row_inrange = row_less and row_more

            col_more = col > ground_truth_start[1] - car_map_origin[1]
            col_less = col < ground_truth_start[1] + car_map.shape[1]-car_map_origin[1]
            col_inrange = col_more and col_less

            flag = row_inrange and col_inrange

            return flag
        
        def car_coordinates(row, col):
            """Returns coordinates in the car map"""
            car_row = row - (ground_truth_start[0] - car_map_origin[0])
            car_col = col - (ground_truth_start[1] - car_map_origin[1])

            return car_row, car_col

        reward_map = np.zeros(reward_map_shape)

        # Assign rewards to unmapped area
        ground_truth = self.simulator.simulated_map.map
        for row in range(reward_map_shape[0]):
            for col in range(reward_map_shape[1]):
                if is_in_map(row, col):
                    car_row, car_col = car_coordinates(row, col)
                    if car_map[car_row, car_col] == -1:
                        rand_choice = np.random.choice([0,self.empty_reward], 1, p=[1 - probability, probability])
                        reward_map[row, col] = rand_choice
                    else:
                        rand_choice = np.random.choice([0,self.negative_reinforcement], 1, p=[1 - probability, probability])
                        reward_map[row, col] = rand_choice
                elif ground_truth[row, col] == 0:
                    rand_choice = np.random.choice([0,self.empty_reward], 1, p=[1 - probability, probability])
                    reward_map[row, col] = rand_choice
                elif ground_truth[row, col] == 1:
                    rand_choice = np.random.choice([0,self.obstacle_reward], 1, p=[1 - probability, probability])
                    reward_map[row, col] = rand_choice

        self.reward_map = reward_map

    def collect_sparse(self, done, sim: simulator.Simulator):
        """
        Collects reward from the map
        
        args:
            done (bool): True if it is the end of an episode
            sim (Simulator object): simulator that we are in right now
        """
        self.update_simulator(sim)

        if done:
            return -500

        global_coords = self.simulator.get_global_coordinates()
        lidar_radius = self.simulator.car.lidar_radius
        
        reward = 0
        # Loop through indices and record ones that are within radius
        for row in np.arange(global_coords[0]-lidar_radius, global_coords[0]+lidar_radius+1):
            for col in np.arange(global_coords[1]-lidar_radius, global_coords[1]+lidar_radius+1):
                distance = ((row-global_coords[0])**2 + (col-global_coords[1])**2)**0.5
                if distance <= lidar_radius:
                    reward += self.reward_map[row, col]

        return reward