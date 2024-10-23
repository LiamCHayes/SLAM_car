"""
Calculate the reward for the RL network
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
        simulator (Simulator object): simulator that we are in right now
        empty_reward (int): amount of reward for discovering an empty space
        obstacle_reward (int): amount of reward for discovering an obstacle space

    attributes:
        simulator (Simulator object): simulator that we are in right now
        empty_reward (int): amount of reward for discovering an empty space
        obstacle_reward (int): amount of reward for discovering an obstacle space
    """
    def __init__(self, sim: simulator.Simulator, empty_reward, obstacle_reward):
        self.simulator = sim
        self.empty_reward = empty_reward
        self.obstacle_reward = obstacle_reward

    def discover_reward(self):
        """
        Computes the map of rewards for the whole map. The car collects 
        the reward by reading the space with its LIDAR.

        Given a the area mapped out by the robot and the ground truth map of
        the environment, compute the map of the rewards obtainable by the robot.

        returns:
            reward_map (numpy array): array of size self.simulator.simulated_map.size 
                                      that represents the rewards obtainable
        """
        # Variables we need to find the unmapped area
        car_map = self.simulator.car.map
        car_map_origin = self.simulator.car.np_origin_coordinates
        ground_truth_start = self.simulator.path[:, 0]

        car_map_size = car_map.size
        reward_size = self.simulator.simulated_map.map.size

        def is_in_map(row, col):
            row_more = row > ground_truth_start[0] - car_map_origin[0]
            row_less = row < ground_truth_start[0] + car_map.size[0]-car_map_origin[0]
            row_inrange = row_less and row_more

            col_more = col > ground_truth_start[1] - car_map_origin[1]
            col_less = col < ground_truth_start[1] + car_map.size[1]-car_map_origin[1]
            col_inrange = col_more and col_less

            flag = row_inrange and col_inrange

            return flag

        reward = np.zeros(reward_size)

        # Assign rewards to unmapped area
        ground_truth = self.simulator.simulated_map.map
        for row in reward_size[0]:
            for col in reward_size[1]:
                # compute the car reading of this spot (if it exists)
                car_map_idx = None
                if is_in_map(row, col):
                    
                    pass

                # Assign reward

        return reward
