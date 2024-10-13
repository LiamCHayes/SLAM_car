# SLAM Car
The goal is a car that can autonomously navigate an enclosed unknown environment, map the environment, and possibly return to the start point. 
The car will use LIDAR, machine learning, and NN for waypoint generation and mapping. All real-time compute will be local on an Espressif ESP32-S3 Dev 
microcontroller.

## Scene Representation
Since we are mapping out a 2D plane, we can represent the space with a numpy array. The map will be inscribed in a NxM numpy array where -1 represents an unmapped
area, 0 represents a free space, and a positive number represents an obstacle. More generally, each element in the array will either be a probability 
that an obstacle is there or a -1.

## Data Flow
Initialize the map with the origin at the start point.

