"""
Trains a discrete pytorch deep q model
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
import deep_q_networks

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resume", action="store_true", help="Whether to resume training or start a new training cycle")
    parser.add_argument("-p", "--pick_up_from", type=str, help="Path to resume from. Format path/to/directory")
    parser.add_argument("-s", "--save_path", type=str, help="Path to save the networks and metrics to. Format path/to/directory")
    args = parser.parse_args()
    return args

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

def get_pink_noise_discrete(noise_rl, noise_qd):
    """
    Discretize two pink noise gaussians to decide what quadrant to move to in exploration
    
    args:
        noise_rl, noise_qd (float): independent samples from a pink noise gaussian distribution
    """
    if noise_rl[step] >= 0:
        # On a clock: 12, 1:30, 3, 4:30
        noise_idx = np.arange(8)
    else: 
        # On a clock: 6, 7:30, 8, 9:30
        noise_idx = np.arange(8, 16)
    if noise_qd[step] >= 0:
        # Choose first quadrant or third quadrant depending on noise_rl
        noise_idx = noise_idx[:4]
    else:
        # Choose second quadrant or fourth quadrant depending on noise_rl
        noise_idx = noise_idx[4:]

    return noise_idx

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

if __name__== "__main__":
    args = parse_args()

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
    target_update_freq = 100
    save_freq = 10
    gamma = 0.95
    epsilon = 0.75

    # Networks
    """ DQ1 action set
    act_mag = np.floor(lidar_radius * 0.75)
    actions = [(act_mag, 0), (0, act_mag), (-act_mag, 0), (0, -act_mag),
               (act_mag, act_mag), (-act_mag, act_mag), (act_mag, -act_mag), (-act_mag, -act_mag)]
    """
    # DQ2 action set
    act_mag_big = np.floor(lidar_radius * 0.8)
    act_mag_small = np.floor(lidar_radius * 0.25)
    # Actions go around the circle like a clock
    actions = [(0, act_mag_big), (0, act_mag_small), (act_mag_big, act_mag_big), (act_mag_small, act_mag_small),
               (act_mag_big, 0), (act_mag_small, 0), (act_mag_big, -act_mag_big), (act_mag_small, -act_mag_small),
               (0, -act_mag_big), (0, -act_mag_small), (-act_mag_big, -act_mag_big), (-act_mag_small, -act_mag_small),
               (-act_mag_big, 0), (-act_mag_small, 0), (-act_mag_big, act_mag_big), (-act_mag_small, act_mag_small)]
    action_size = len(actions)

    if args.resume:
        policy_net = torch.load(f'{args.pick_up_from}/policy_net.pth').to(device)
        target_net1 = torch.load(f'{args.pick_up_from}/target_net1.pth').to(device)
        # target_net2 = torch.load(f'{args.pick_up_from}/target_net2.pth').to(device)
    else:
        policy_net = deep_q_networks.DeepQ(action_size).to(device)
        target_net1 = deep_q_networks.DeepQ(action_size).to(device)
        target_net1.load_state_dict(policy_net.state_dict())
        target_net2 = deep_q_networks.DeepQ(action_size).to(device)
        target_net2.load_state_dict(policy_net.state_dict())

    # Optimizers
    lr = 1e-4
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    # Replay Buffer
    if args.resume:
        # Load the replay buffer
        with open(f'{args.pick_up_from}/memory.pkl', 'rb') as file:
            memory = pickle.load(file)

        # Training loop
        tot_reward_list = np.load(f"{args.pick_up_from}/tot_reward.npy")
        pct_explored_list = np.load(f"{args.pick_up_from}/pct_explored.npy")
        episode_len_list = np.load(f"{args.pick_up_from}/episode_len.npy")
        loss_list = np.load(f"{args.pick_up_from}/tot_loss.npy")
    else:
        memory = ReplayBuffer(memory_capacity)
        # Fill the replay buffer with actions
        while memory.__len__() < batch_size:
            done = False
            sim = reset_env(map_size, lidar_radius)

            while not done and memory.__len__() < batch_size:
                curr_state = sim.car.lidar_reading
                reward_map = rewarder.discover_reward(sim)

                # Choose random action and go with it
                action_selection = np.random.randint(8)
                action = actions[action_selection]

                no_collision, next_state = sim.step(action, False)
                
                done = not no_collision
                reward = rewarder.collect_reward(done, sim)

                memory.add(curr_state, action_selection, reward, next_state, done)

        # Training loop
        tot_reward_list = np.array([])
        pct_explored_list = np.array([])
        episode_len_list = np.array([])
        loss_list = np.array([])

    # Training
    for episode in range(n_episodes):
        # Episode loop variables
        step = 0
        done = False
        sim = reset_env(map_size, lidar_radius)
        noise_rl = cn.powerlaw_psd_gaussian(1, episode_len)
        noise_qd = cn.powerlaw_psd_gaussian(1, episode_len)

        # Episode metrics
        total_reward = 0
        total_loss = 0

        while not done and step < episode_len:
            # Get state and reward map
            curr_state = sim.car.lidar_reading
            state = np_to_tensor(curr_state).unsqueeze(0).to(device)
            reward_map = rewarder.discover_sparse(sim, probability=0.1)

            # Get noise for action (pretty involved because it is colored discrete noise)
            noise_idx = get_pink_noise_discrete(noise_rl, noise_qd)

            # Get action probabilities
            action_probs = policy_net.forward(state)

            # Select exploration or exploitation action
            action_selection = [torch.argmax(action_probs).item(), np.random.choice(noise_idx)]
            action_selection = np.random.choice(action_selection, 1, p=[1-epsilon, epsilon])
            act = actions[action_selection]
            
            # Execute! Get reward and done bool
            no_collision, next_state = sim.step(act, False)

            done = not no_collision
            reward = rewarder.collect_sparse(done, sim)

            # Record in memory
            memory.add(curr_state, action_selection, reward, next_state, done)

            # Sample memory
            batch = memory.sample(batch_size)

            # Update
            current_q = policy_net.forward(batch[0])
            col_idxs = batch[1].type(torch.int)
            row_idxs = torch.arange(batch[1].size(0)).to(device)
            current_q = current_q[row_idxs, col_idxs]

            if done:
                target_q = batch[2]
            else:
                target_q = target_net1.forward(batch[3])
                col_idxs = torch.argmax(target_q, dim=1)    
                row_idxs = torch.arange(target_q.size(0)).to(device)
                target_q = batch[2] + gamma * target_q[row_idxs, col_idxs]

            loss = torch.nn.functional.mse_loss(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
            optimizer.step()

            # Update episode loop stats
            step += 1
            total_reward += reward
            total_loss += loss.detach().cpu().numpy()

        # Store and print episode stats
        pct_explored = (sim.car.map >= 0).sum() / (sim.simulated_map.map >= 0).sum()
        pct_explored_list = np.append(pct_explored_list, pct_explored)
        tot_reward_list = np.append(tot_reward_list, total_reward)
        episode_len_list = np.append(episode_len_list, step)
        loss_list = np.append(loss_list, total_loss)

        print("\n")
        print("Episode number: ", episode)
        print("Episode length: ", step)
        print("Total reward: ", total_reward)
        print("Total loss: ", total_loss)
        print("Percent explored: ", round(pct_explored * 100, 2), "%")

        # Update target networks 
        if episode % target_update_freq == 0:
            # Update target networks
            target_net1.load_state_dict(policy_net.state_dict())
            #target_net2.load_state_dict(policy_net.state_dict())

        # Save networks and metrics
        if episode % save_freq == 0:
            # Save networks
            torch.save(policy_net, f"{args.save_path}/policy_net.pth")
            torch.save(target_net1, f"{args.save_path}/target_net1.pth")
            #torch.save(target_net2, f"{args.save_path}/target_net2.pth")

            # Save loop vals
            np.save(f"{args.save_path}/pct_explored.npy", pct_explored_list)
            np.save(f"{args.save_path}/tot_reward.npy", tot_reward_list)
            np.save(f"{args.save_path}/episode_len.npy", episode_len_list)
            np.save(f"{args.save_path}/tot_loss.npy", loss_list)

            # Save replay buffer
            with open(f'{args.save_path}/memory.pkl', 'wb') as outp:
                pickle.dump(memory, outp)

            # Save reward
            with open(f'{args.save_path}/reward.pkl', 'wb') as outp:
                pickle.dump(rewarder, outp)
