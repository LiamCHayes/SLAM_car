# SLAM Car
The goal is a car that can autonomously navigate an enclosed unknown environment, map the environment, and possibly return to the start point. The car will use LIDAR machine learning, and NN for waypoint generation and mapping. All real-time compute will be local on an Raspberry Pi 4.

## Scene Representation
Since we are mapping out a 2D plane, we can represent the space with a numpy array. The map will be inscribed in a NxM numpy array where -1 represents an unmapped
area, 0 represents a free space, and a positive number represents an obstacle. More generally, each element in the array will either be a probability that an obstacle is there or a -1.

## Program Flow
0. Initialize the map with the first LIDAR observation. Initialize world coordinates with the origin at the start point, x as longitudinal axis, and y as lateral axis.

While there is still unmapped space:
1. Feed forward NN inference to generate a waypoint based on the known observations.
2. Move to the waypoint.
3. New LIDAR observation.
4. State estimation to determine new coordinates.
5. Rotate and translate new LIDAR observation to the world perspective. 
6. Append observations to the map.

Considerations:

The numpy array will grow as we map out more area. Should we transform the world coordinates to minimize this growth with new observations?

As the number of steps increase, errors can accumulate. How are we doing state estimation and dealing with errors?

The resulting map should be a probability desity function (PDF) $f(x, y)$ representing the probability an obstacle is at the coordinates $(x,y)$. To do this, in step 6 we should increment each element by 1 where an obstacle is observed. We should also save the number of observations each element has in a seperate array. By dividing the map by the total observation array, we can obtain probabilities of where obstacles are for the whole map.


