"""
Unit tests 
"""

#######
# Setup
#######
import numpy as np
import matplotlib.pyplot as plt

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


######
# Main
######
if __name__ == '__main__':
    simulator_test()
