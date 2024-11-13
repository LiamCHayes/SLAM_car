"""
Shows training metrics and runs the model on a new environment
"""

#######
# Setup
#######
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import torch
import argparse

import simulator
import losses
from train_SAC import np_to_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, help="Path to model to evaluate. Format path/to/directory")
parser.add_argument("-r", "--results", action="store_true", help="Show results")
parser.add_argument("-t", "--test", action="store_true", help="Show test")
args = parser.parse_args()


###########
# Functions
############
def visualize_results(path):
    """
    Visualize training metrics
    """
    ep_len = np.load(f"{path}/episode_len.npy")
    pct_explored = np.load(f"{path}/pct_explored.npy") * 100
    tot_reward = np.load(f"{path}/tot_reward.npy")

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

def test_model(path):
    lidar_radius = 50
    rewarder = losses.Reward(empty_reward=5, 
                             obstacle_reward=0, 
                             negative_reinforcement=-1)

    # Create map
    sim_map = simulator.SimulatedMap(size=(320, 320))
    sim_map.create_map()
    sim_map.create_obstacles(np.random.randint(4, 15))

    sim = simulator.Simulator(sim_map)
    sim.spawn_car(lidar_radius, plot=True)

    model = torch.load(f'{path}/actor_ntw.pth').to(device)
    model.eval()  # Set the model to evaluation mode
    
    total_reward = 0
    no_collision = True
    while no_collision:
        curr_state = sim.car.lidar_reading
        state = np_to_tensor(curr_state).unsqueeze(0).to(device)
        reward_map = rewarder.discover_reward(sim)

        # Get action 
        action, _ = model.sample(state)

        # Execute! Get reward and done bool
        no_collision, next_state = sim.step(action, False, plot=True)
        total_reward += rewarder.collect_reward(not no_collision, sim)
    
        print("Action: ", action)
    print("Total reward: ", total_reward)
    

######
# Main
######
if __name__ == '__main__':
    if args.results:
        visualize_results(args.path)
    if args.test:
        test_model(args.path)