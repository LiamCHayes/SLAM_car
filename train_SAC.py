"""
Trains the pytorch SAC model
"""

#######
# Setup
#######
import torch
import numpy as np
from collections import deque
import random
import pickle
import argparse
import colorednoise as cn

import simulator
import losses
import SAC_networks

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resume", action="store_true", help="Whether to resume training or start a new training cycle")
parser.add_argument("-p", "--pick_up_from", type=str, help="Path to resume from. Format path/to/directory")
parser.add_argument("-s", "--save_path", type=str, help="Path to save the networks and metrics to. Format path/to/directory")
args = parser.parse_args()

####################
# Utils for training
####################
def create_env(size, lidar_radius):
    """
    creates the environment

    args: 
        size (tupple): size of the map
        lidar_radius (int): range of the lidar

    returns:
        sim (Simulator object): new environment
    """
    # Define
    sim_map = simulator.SimulatedMap(size)
    sim_map.create_map()
    sim_map.create_obstacles(np.random.randint(4, 15))

    # Create simulator
    sim = simulator.Simulator(sim_map)
    
    # Spawn car
    sim.spawn_car(lidar_radius)

    return sim

def reset_env(size, lidar_radius):
    sim = create_env(size, lidar_radius)
    return sim

def np_to_tensor(arr: np.ndarray):
    """
    Converts numpy array to the right format of tensor
    """
    arr = arr.astype(np.float32)
    tensor = torch.from_numpy(arr)
    tensor = tensor.unsqueeze(0)
    return tensor

class ReplayBuffer:
    """
    Replay Buffer. Stores (state, action, reward, next_state, done)

    args:
        capacity (int): buffer capacity

    attributes:
        buffer (deque): the memory
        capacity (int): buffer capacity    
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        # Give states a channel to input into the network
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        # Make all actions the same shape
        if isinstance(action, torch.Tensor):
            action = action.cpu()
            action = action.detach().numpy()[0]
            action = action.astype(np.int32)
            action = tuple(action)
        
        # Add to buffer
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from the buffer.
        
        Returns:
            a batch of (state, action, reward, next_state, done)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)

        # Convert to PyTorch tensors
        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.float32).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device)
        )

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

###################
# Define parameters
###################
# Environment
lidar_radius = 50
map_size = (500, 500)
if args.resume:
    with open(f'{args.pick_up_from}/reward.pkl', 'rb') as file:
        rewarder = pickle.load(file)
else:
    rewarder = losses.Reward(empty_reward=5, 
                            obstacle_reward=-3, 
                            negative_reinforcement=-1)

# Training params
n_episodes = 10000
episode_len = 25
batch_size = 32
memory_capacity = 1000
gamma = 0.95
alpha = 0.2
target_update_freq = 100

# Networks
if args.resume:
    actor_ntw = torch.load(f'{args.pick_up_from}/actor_ntw.pth').to(device)
    critic_ntw1 = torch.load(f'{args.pick_up_from}/critic_ntw1.pth').to(device)
    critic_ntw2 = torch.load(f'{args.pick_up_from}/critic_ntw2.pth').to(device)
    target_critic_ntw1 = torch.load(f'{args.pick_up_from}/target_critic_ntw1.pth').to(device)
    target_critic_ntw2 = torch.load(f'{args.pick_up_from}/target_critic_ntw2.pth').to(device)
else:
    actor_ntw = SAC_networks.Actor(lidar_radius).to(device)
    critic_ntw1 = SAC_networks.Critic().to(device)
    critic_ntw2 = SAC_networks.Critic().to(device)
    target_critic_ntw1 = SAC_networks.Critic().to(device)
    target_critic_ntw1.load_state_dict(critic_ntw1.state_dict())
    target_critic_ntw2 = SAC_networks.Critic().to(device)
    target_critic_ntw2.load_state_dict(critic_ntw2.state_dict())

# Optimizers
lr = 0.001
optimizer_A = torch.optim.Adam(actor_ntw.parameters(), lr=lr)
optimizer_C1 = torch.optim.Adam(critic_ntw1.parameters(), lr=lr)
optimizer_C2 = torch.optim.Adam(critic_ntw2.parameters(), lr=lr)

# Replay Buffer
memory = ReplayBuffer(memory_capacity)

##########
# Training
##########
if __name__== "__main__":
    # Resume training or start from new
    if args.resume:
        # Load the replay buffer
        with open(f'{args.pick_up_from}/memory.pkl', 'rb') as file:
            memory = pickle.load(file)

        # Training loop
        tot_reward_list = np.load(f"{args.pick_up_from}/tot_reward.npy")
        pct_explored_list = np.load(f"{args.pick_up_from}/pct_explored.npy")
        episode_len_list = np.load(f"{args.pick_up_from}/episode_len.npy")
        print(tot_reward_list)
    else:
        # Fill the replay buffer with actions
        while memory.__len__() < batch_size:
            done = False
            sim = reset_env(map_size, lidar_radius)

            while not done and memory.__len__() < batch_size:
                curr_state = sim.car.lidar_reading
                reward_map = rewarder.discover_reward(sim)

                # Choose random action and go with it
                row_move = random.randint(-lidar_radius, lidar_radius)
                col_move = random.randint(-lidar_radius, lidar_radius)
                action = (row_move, col_move)

                no_collision, next_state = sim.step(action, False)
                
                done = not no_collision
                reward = rewarder.collect_reward(done, sim)

                memory.add(curr_state, action, reward, next_state, done)

        # Training loop
        tot_reward_list = np.array([])
        pct_explored_list = np.array([])
        episode_len_list = np.array([])

    for episode in range(n_episodes):
        # Episode loop variables
        step = 0
        done = False
        sim = reset_env(map_size, lidar_radius)

        # Episode metrics
        total_reward = 0

        while not done and step < episode_len:
            # Get state and reward map
            curr_state = sim.car.lidar_reading
            state = np_to_tensor(curr_state).unsqueeze(0).to(device)
            reward_map = rewarder.discover_reward(sim)

            # Get action 
            noise = cn.powerlaw_psd_gaussian(1, episode_len)
            action, log_prob = actor_ntw.sample_pink(state, noise[step])

            # Execute! Get reward and done bool
            no_collision, next_state = sim.step(action, False)

            done = not no_collision
            reward = rewarder.collect_reward(done, sim)

            # Record in memory
            memory.add(curr_state, action, reward, next_state, done)

            # Sample memory
            batch = memory.sample(batch_size)
            
            # Update critics
            with torch.no_grad():
                next_action, next_log_prob = actor_ntw.sample(batch[3])
                target_q1 = target_critic_ntw1(batch[3], next_action)
                target_q2 = target_critic_ntw2(batch[3], next_action)
                target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                target_q = batch[2] + (1 - batch[4]) * gamma * target_q
            
            current_q1 = critic_ntw1(batch[0], batch[1])
            current_q2 = critic_ntw2(batch[0], batch[1])
            
            critic1_loss = torch.nn.functional.mse_loss(current_q1, target_q)
            critic2_loss = torch.nn.functional.mse_loss(current_q2, target_q)

            optimizer_C1.zero_grad()
            critic1_loss.backward()
            optimizer_C1.step()
            
            optimizer_C2.zero_grad()
            critic2_loss.backward()
            optimizer_C2.step()

            # Update actor
            action, log_prob = actor_ntw.sample(batch[0])
            q1 = critic_ntw1(batch[0], action)
            q2 = critic_ntw2(batch[0], action)
            q = torch.min(q1, q2)
            actor_loss = (alpha * log_prob - q).mean()
            
            optimizer_A.zero_grad()
            actor_loss.backward()
            optimizer_A.step()

            # Update episode loop stats
            step += 1
            total_reward += reward
        
        # Store and print episode stats
        pct_explored = (sim.car.map >= 0).sum() / (sim.simulated_map.map >= 0).sum()
        pct_explored_list = np.append(pct_explored_list, pct_explored)
        tot_reward_list = np.append(tot_reward_list, total_reward)
        episode_len_list = np.append(episode_len_list, step)

        print("\n")
        print("Episode length: ", step)
        print("Total reward: ", total_reward)
        print("Percent explored: ", round(pct_explored * 100, 2), "%")
        
        # Update target networks and save networks once in a while
        if episode % target_update_freq == 0:
            # Update target networks
            target_critic_ntw1.load_state_dict(critic_ntw1.state_dict())
            target_critic_ntw2.load_state_dict(critic_ntw2.state_dict())

            # Save networks
            torch.save(actor_ntw, f"{args.save_path}/actor_ntw.pth")
            torch.save(critic_ntw1, f"{args.save_path}/critic_ntw1.pth")
            torch.save(critic_ntw2, f"{args.save_path}/critic_ntw2.pth")
            torch.save(target_critic_ntw1, f"{args.save_path}/target_critic_ntw1.pth")
            torch.save(target_critic_ntw2, f"{args.save_path}/target_critic_ntw2.pth")

            # Save loop vals
            np.save(f"{args.save_path}/pct_explored.npy", pct_explored_list)
            np.save(f"{args.save_path}/tot_reward.npy", tot_reward_list)
            np.save(f"{args.save_path}/episode_len.npy", episode_len_list)

            # Save replay buffer
            with open(f'{args.save_path}/memory.pkl', 'wb') as outp:
                pickle.dump(memory, outp)

            # Save reward
            with open(f'{args.save_path}/reward.pkl', 'wb') as outp:
                pickle.dump(rewarder, outp)
