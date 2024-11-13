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
        simulator = simulator.SimulatedMap(size=(320, 320))
        simulator.create_map()
        simulator.create_obstacles(np.random.randint(4, 15))
        plt.imshow(simulator.map)
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

    sim.spawn_car(lidar_radius=15, plot=True)
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
        loss.discover_reward(sim)
        no_collision = sim.step((x,y), False, plot=True)
        done = not no_collision
        reward_val = loss.collect_reward(done, sim)
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
    

######
# Main
######
if __name__ == '__main__':
    visualize_results()
    test_model()
