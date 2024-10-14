"""
Unit tests 
"""

#######
# Setup
#######
import numpy as np
import matplotlib.pyplot as plt

import simulate_map


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


######
# Main
######
if __name__ == '__main__':
    simulate_map_test()
