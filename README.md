# SLAM Car
The goal is a car that can autonomously navigate an enclosed unknown environment, map the environment, and possibly return to the start point. The car will use LiDAR machine learning, and NN for waypoint generation and mapping. All real-time compute will be local on an Raspberry Pi 4.

## Scene Representation
Since we are mapping out a 2D plane, we can represent the space with a numpy array. The map will be inscribed in a NxM numpy array where -1 represents an unmapped
area, 0 represents a free space, and a 1 number represents an obstacle. 

## Program Flow
0. Initialize the map with the first LIDAR observation. Initialize world coordinates with the origin at the start point, x as longitudinal axis, and y as lateral axis.

While there is still unmapped space:
1. Feed forward NN inference to generate a waypoint based on the known observations.
2. Move to the waypoint.
3. New LIDAR observation.
4. State estimation to determine new coordinates.
5. Rotate and translate new LIDAR observation to the world perspective. 
6. Append observations to the map.

## How To Use
To train your own networks, use any of the files named train_*.py. We have implemented soft actor critic, classic deep Q learning, deep Q learning with a limited action set, and deep Q learning with a long short term memory network. For our paper, we found that the deep Q learning with a limited action set gave the best results. Any file named *_networks.py contains the network architectures that we used, implemented in PyTorch. To evaluate the networks, we have evaluate.py and get_action.py. evaluate.py evaluates the networks in the simulator, and get_action.py interfaces with our ROS stack to evaluate the networks on our robot.

Our simulator where we train the models is entireley contained in simulator.py.




