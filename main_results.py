"""
Produces all of the python plots
For the Exploding Loss Plot (Figure 6) look at the file analysis.r
All of the plots are randomly generated, so it will not produce 
the exact same plots but it is the same data generating process
"""

import simulator
import matplotlib.pyplot as plt
import numpy as np

def simulate_map_test():
    """
    Does a sample of 10 randomly generated ground truth maps
    """
    print("Generating 4 maps...")
    for i in range(4):
        print(f'Map {i + 1} generated')
        sim = simulator.SimulatedMap(size=(320, 320))
        sim.create_map()
        sim.create_obstacles(np.random.randint(4, 15))
        plt.imshow(sim.map)
        plt.show()

def simulator_test():
    """
    Initializes simulator, performs and plots 3 steps
    """
    print("\nStarting simulator...")
    # Create map
    sim_map = simulator.SimulatedMap(size=(320, 320))
    sim_map.create_map()
    sim_map.create_obstacles(np.random.randint(4, 15))

    sim = simulator.Simulator(sim_map)

    sim.spawn_car(lidar_radius=30, plot=True)
    no_collision = True
    while no_collision:
        x = int(input('Row move: '))
        y = int(input('col move: '))
        no_collision = sim.step((x,y), False, plot=True)
        plt.show()

if __name__ == "__main__":
    simulate_map_test()
    simulator_test()