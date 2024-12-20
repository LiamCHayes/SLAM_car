"""
Unit tests 
"""

#######
# Setup
#######
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import torch
import colorednoise as cn
import pandas as pd
import time
from deep_q_networks import DeepQ

import simulator
import losses
from train_SAC import np_to_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#######
# Tests
#######
def simulate_map_test():
    """
    Does a sample of 10 randomly generated ground truth maps
    """
    for i in range(10):
        print(f'Map {i + 1} generated')
        sim = simulator.SimulatedMap(size=(320, 320))
        sim.create_map()
        sim.create_obstacles(np.random.randint(4, 15))
        plt.imshow(sim.map)
        plt.show()

def simulator_test():
    """
    Initializes simulator, performs and plots 3 steps
    """
    # Create map
    sim_map = simulator.SimulatedMap(size=(320, 320))
    sim_map.create_map()
    sim_map.create_obstacles(np.random.randint(4, 15))

    sim = simulator.Simulator(sim_map)

    sim.spawn_car(lidar_radius=30, plot=True)
    no_collision = True
    while no_collision:
        x = int(input('Row move: '))
        y = int(input('col move: '))
        no_collision = sim.step((x,y), False, plot=True)
        plt.show()

def reward_test():
    """
    Test reward functions
    """
    # Create map
    sim_map = simulator.SimulatedMap(size=(320, 320))
    sim_map.create_map()
    sim_map.create_obstacles(np.random.randint(4, 15))

    sim = simulator.Simulator(sim_map)
    loss = losses.Reward(1, 2, -1)

    sim.spawn_car(lidar_radius=15, plot=True)
    no_collision = True
    while no_collision:
        x = int(input('Row move: '))
        y = int(input('col move: '))
        loss.discover_sparse(sim, 0.001)
        no_collision = sim.step((x,y), False, plot=True)
        done = not no_collision
        reward_val = loss.collect_sparse(done, sim)
        print(reward_val)
        plt.imshow(loss.reward_map)
        plt.show()

def visualize_results():
    """
    Visualize training metrics
    """
    ep_len = np.load("models/SAC/episode_len.npy")
    pct_explored = np.load("models/SAC/pct_explored.npy") * 100
    tot_reward = np.load("models/SAC/tot_reward.npy")

    num_timesteps = ep_len.size

    plt.plot(np.arange(num_timesteps), tot_reward, label='Reward')
    plt.legend()
    plt.title("Reward")
    plt.show()

    plt.plot(np.arange(num_timesteps), ep_len, label='Episode_length')
    plt.plot(np.arange(num_timesteps), pct_explored, label='Percent_area_explored')
    plt.legend()
    plt.title("Episode Length and Percent Explored")
    plt.show()

def test_model():
    lidar_radius = 50

    # Create map
    sim_map = simulator.SimulatedMap(size=(320, 320))
    sim_map.create_map()
    sim_map.create_obstacles(np.random.randint(4, 15))

    sim = simulator.Simulator(sim_map)
    sim.spawn_car(lidar_radius, plot=True)

    model = torch.load('models/SAC/actor_ntw.pth').to(device)
    model.eval()  # Set the model to evaluation mode
    
    init_reading = sim.car.map
    no_collision = True
    while no_collision:
        curr_state = sim.car.lidar_reading
        state = np_to_tensor(curr_state).unsqueeze(0).to(device)

        # Get action 
        action, log_prob = model.sample(state)

        # Execute! Get reward and done bool
        no_collision, next_state = sim.step(action, False, plot=True)
    
def test_discrete_noise():
    episode_len = 25
    noise_rl = cn.powerlaw_psd_gaussian(1, episode_len)
    noise_ud = cn.powerlaw_psd_gaussian(1, episode_len)
    
    for step in range(episode_len):
        if noise_rl[step] >= 0:
            # On a clock: 12, 1:30, 3, 4:30
            noise_idx = np.arange(8)
        else: 
            # On a clock: 6, 7:30, 8, 9:30
            noise_idx = np.arange(8, 16)
        if noise_ud[step] >= 0:
            noise_idx = noise_idx[:4]
        else:
            noise_idx = noise_idx[4:]
        print("noise_rl: ", noise_rl)
        print("noise_ud: ", noise_ud)
        print(noise_idx)
        print(np.random.choice(noise_idx))
        input("")

def get_data(data_path, save_path):
    ep_len = np.load(f"{data_path}/episode_len.npy")
    pct_explored = np.load(f"{data_path}/pct_explored.npy") * 100
    tot_reward = np.load(f"{data_path}/tot_reward.npy")
    loss = np.load(f'{data_path}/tot_loss.npy')
    data = {'loss' : loss, 'reward' : tot_reward, 'pct_explored' : pct_explored, 'ep_len' : ep_len}
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=True)

def get_polar_matrices(np_origin):
    r_matrix = np.zeros((np_origin*2+1, np_origin*2+1))
    theta_matrix = np.zeros((np_origin*2+1, np_origin*2+1))
    for row in range(r_matrix.shape[0]):
        for col in range(r_matrix.shape[1]):
            x = col - np_origin
            y = -(row - np_origin)
            r_matrix[row, col] = round((x**2 + y**2) ** 0.5)
            if x != 0:
                if x > 0 and y > 0:
                    theta_matrix[row, col] = np.arctan(y/x)
                elif x < 0:
                    theta_matrix[row, col] = np.arctan(y/x) + np.pi
                elif x > 0 and y < 0:
                    theta_matrix[row, col] = np.arctan(y/x) + np.pi * 2
            else:
                if y > 0:
                    theta_matrix[row, col] = np.pi/2
                elif y < 0:
                    theta_matrix[row, col] = 3*np.pi/2
    
    return r_matrix, theta_matrix

######
# Main
######
if __name__ == '__main__':
    simulate_map_test()