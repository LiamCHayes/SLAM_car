"""
Unit tests 
"""

#######
# Setup
#######
import numpy as np
import matplotlib.pyplot as plt

import simulate_map
import car
import simulator


#######
# Tests
#######
def simulate_map_test():
    """
    Does a sample of 10 randomly generated ground truth maps
    """
    for i in range(10):
        print(f'Map {i + 1} generated')
        simulator = simulate_map.SimulatedMap(size=(320, 320))
        simulator.create_map()
        simulator.create_obstacles(np.random.randint(4, 15))
        plt.imshow(simulator.map)
        plt.show()

def spawn_car_test():
    # Create map
    sim_map = simulate_map.SimulatedMap(size=(320, 320))
    sim_map.create_map()
    sim_map.create_obstacles(np.random.randint(4, 15))

    sim = simulator.Simulator(sim_map)

    sim.spawn_car(lidar_radius=20, plot=True)


######
# Main
######
if __name__ == '__main__':
    spawn_car_test()
