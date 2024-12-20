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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path to model to evaluate. Format path/to/directory")
    parser.add_argument("-m", "--model", type=str, help="Model number. 1 sac, 2 dqn, 3 lstm, 4 dqn limited")
    parser.add_argument("-r", "--results", action="store_true", help="Show results")
    parser.add_argument("-t", "--test", action="store_true", help="Show test")
    args = parser.parse_args()
    return args


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
    plt.legend()
    plt.title("Episode Length")
    plt.show()

    plt.plot(np.arange(num_timesteps), pct_explored, label='Percent_area_explored')
    plt.legend()
    plt.title("Percent Explored")
    plt.show()

def test_model(path):
    lidar_radius = 50
    rewarder = losses.Reward(empty_reward=5, 
                             obstacle_reward=0, 
                             negative_reinforcement=-1)

    model = torch.load(f'{path}/actor_ntw.pth').to(device)
    model.eval()  # Set the model to evaluation mode

    while True:

        # Create map
        sim_map = simulator.SimulatedMap(size=(320, 320))
        sim_map.create_map()
        sim_map.create_obstacles(np.random.randint(4, 15))

        sim = simulator.Simulator(sim_map)
        sim.spawn_car(lidar_radius, plot=True)
        
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
        
def test_model_dqn(path):
    lidar_radius = 50
    rewarder = losses.Reward(empty_reward=5, 
                             obstacle_reward=0, 
                             negative_reinforcement=-1)

    model = torch.load(f'{path}/policy_net.pth').to(device)
    model.eval()  # Set the model to evaluation mode

    while True:
        # Create map
        sim_map = simulator.SimulatedMap(size=(320, 320))
        sim_map.create_map()
        sim_map.create_obstacles(np.random.randint(4, 15))

        sim = simulator.Simulator(sim_map)
        sim.spawn_car(lidar_radius, plot=True)
    
        total_reward = 0
        no_collision = True
        while no_collision:
            curr_state = sim.car.lidar_reading
            state = np_to_tensor(curr_state).unsqueeze(0).to(device)
            reward_map = rewarder.discover_reward(sim)

            # Get action 
            """act_mag = np.floor(lidar_radius * 0.75)
            actions = [(act_mag, 0), (0, act_mag), (-act_mag, 0), (0, -act_mag),
                (act_mag, act_mag), (-act_mag, act_mag), (act_mag, -act_mag), (-act_mag, -act_mag)]"""
            """act_mag_big = np.floor(lidar_radius * 0.8)
            act_mag_small = np.floor(lidar_radius * 0.25)
            actions = [(0, act_mag_big), (0, act_mag_small), (act_mag_big, act_mag_big), (act_mag_small, act_mag_small),
                (act_mag_big, 0), (act_mag_small, 0), (act_mag_big, -act_mag_big), (act_mag_small, -act_mag_small),
                (0, -act_mag_big), (0, -act_mag_small), (-act_mag_big, -act_mag_big), (-act_mag_small, -act_mag_small),
                (-act_mag_big, 0), (-act_mag_small, 0), (-act_mag_big, act_mag_big), (-act_mag_small, act_mag_small)]"""
            act_mag = np.floor(lidar_radius * 0.75)
            actions = [(0, act_mag), (act_mag, act_mag), (act_mag, 0), (act_mag, -act_mag), 
                (0, -act_mag), (-act_mag, -act_mag), (-act_mag, 0), (-act_mag, act_mag)]
            action_probs = model.forward(state)
            action_selection = torch.argmax(action_probs).item()
            action = actions[action_selection]

            print("Action: ", action)

            # Execute! Get reward and done bool
            no_collision, next_state = sim.step(action, False, plot=True)
            total_reward += rewarder.collect_reward(not no_collision, sim)

        print("Total reward: ", total_reward)

def test_model_lstm(path):
    lidar_radius = 50
    rewarder = losses.Reward(empty_reward=5, 
                             obstacle_reward=0, 
                             negative_reinforcement=-1)

    model = torch.load(f'{path}/policy_net.pth').to(device)
    model.eval()  # Set the model to evaluation mode

    while True:
        # Create map
        sim_map = simulator.SimulatedMap(size=(320, 320))
        sim_map.create_map()
        #sim_map.create_obstacles(np.random.randint(4, 15))

        sim = simulator.Simulator(sim_map)
        sim.spawn_car(lidar_radius, plot=True)
    
        total_reward = 0
        no_collision = True
        while no_collision:
            curr_state = sim.car.lidar_reading
            state = np_to_tensor(curr_state).unsqueeze(0).to(device)
            reward_map = rewarder.discover_reward(sim)

            # Get path in the right format
            path = sim.car.path
            path_tensor = torch.tensor(path, dtype=torch.float32).unsqueeze(0).to(device)

            # Get action 
            """act_mag = np.floor(lidar_radius * 0.75)
            actions = [(act_mag, 0), (0, act_mag), (-act_mag, 0), (0, -act_mag),
                (act_mag, act_mag), (-act_mag, act_mag), (act_mag, -act_mag), (-act_mag, -act_mag)]"""
            """act_mag_big = np.floor(lidar_radius * 0.8)
            act_mag_small = np.floor(lidar_radius * 0.25)
            actions = [(0, act_mag_big), (0, act_mag_small), (act_mag_big, act_mag_big), (act_mag_small, act_mag_small),
                (act_mag_big, 0), (act_mag_small, 0), (act_mag_big, -act_mag_big), (act_mag_small, -act_mag_small),
                (0, -act_mag_big), (0, -act_mag_small), (-act_mag_big, -act_mag_big), (-act_mag_small, -act_mag_small),
                (-act_mag_big, 0), (-act_mag_small, 0), (-act_mag_big, act_mag_big), (-act_mag_small, act_mag_small)]"""
            act_mag = np.floor(lidar_radius * 0.75)
            actions = [(0, act_mag), (act_mag, act_mag), (act_mag, 0), (act_mag, -act_mag), 
                (0, -act_mag), (-act_mag, -act_mag), (-act_mag, 0), (-act_mag, act_mag)]
            action_probs = model.forward(state, path_tensor)
            action_selection = torch.argmax(action_probs).item()
            action = actions[action_selection]

            # Execute! Get reward and done bool
            no_collision, next_state = sim.step(action, False, plot=True)
            total_reward += rewarder.collect_reward(not no_collision, sim)
        
            print("Action: ", action)
        print("Total reward: ", total_reward)

def test_model_dqn_limited(path):
    lidar_radius = 50
    rewarder = losses.Reward(empty_reward=5, 
                             obstacle_reward=0, 
                             negative_reinforcement=-1)

    model = torch.load(f'{path}/policy_net.pth').to(device)
    model.eval()  # Set the model to evaluation mode

    while True:
        # Create map
        sim_map = simulator.SimulatedMap(size=(320, 320))
        sim_map.create_map()
        sim_map.create_obstacles(np.random.randint(4, 15))

        sim = simulator.Simulator(sim_map)
        sim.spawn_car(lidar_radius, plot=True)
    
        total_reward = 0
        no_collision = True
        prev_act_selection = None
        positive_action = True
        while no_collision:
            curr_state = sim.car.lidar_reading
            state = np_to_tensor(curr_state).unsqueeze(0).to(device)
            reward_map = rewarder.discover_reward(sim)

            # Define action set 
            act_mag = np.floor(lidar_radius * 0.75)
            actions = [(0, act_mag), (act_mag, act_mag), (act_mag, 0), (act_mag, -act_mag), 
                (0, -act_mag), (-act_mag, -act_mag), (-act_mag, 0), (-act_mag, act_mag)]
            
            # Get action (exclude the opposite action to avoid going in circles)
            action_probs = model.forward(state)
            if prev_act_selection is not None:
                zero_action = prev_act_selection + 4 if prev_act_selection < 4 else prev_act_selection - 4
                action_probs[0, zero_action] = 0
            action_selection = torch.argmax(action_probs).item()
            action = actions[action_selection]
            prev_act_selection = action_selection

            print("Action: ", action)

            # Execute! Get reward and done bool
            collision = sim.check_collision_future(action)
            while collision:
                print("Collision detected, choosing another action")
                print(action_probs)
                action_probs[0, action_selection] = 0
                action_selection = torch.argmax(action_probs).item()
                if action_probs[0, action_selection] == 0:
                    print("Negative actions activated")
                    positive_action = not positive_action
                    break
                else:
                    action = actions[action_selection]
                collision = sim.check_collision_future(action)

            if positive_action:
                pass
            else:
                neg_act = action_selection + 2 if action_selection < 6 else action_selection - 6
                action = actions[neg_act]

            print(action_probs)

            no_collision, next_state = sim.step(action, False, plot=True)
            total_reward += rewarder.collect_reward(not no_collision, sim)

        print("Total reward: ", total_reward)


######
# Main
######
if __name__ == '__main__':
    args = parse_args()
    if args.results:
        visualize_results(args.path)
    if args.test:
        if int(args.model) == 1:
            test_model(args.path)
        elif int(args.model) == 2:
            test_model_dqn(args.path)
        elif int(args.model) == 3:
            test_model_lstm(args.path)
        elif int(args.model) == 4:
            test_model_dqn_limited(args.path)
            