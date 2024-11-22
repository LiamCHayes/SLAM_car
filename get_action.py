"""
High-level planner

Gets action from the neural net

Put the model in the same directory as this script
"""

import torch

def get_action(lidar_reading, action_magnitude):
    """
    Inputs:
        lidar_reading:
            An nxn numpy array where n is the lidar radius
            0 is a free space and 1 is an obstacle or unknown

            Example where there is a wall in front of the car and lidar radius is 10:
            [[1 1 1 1 1 1 1 1 1 1]
             [1 1 1 1 1 1 1 1 1 1]
             [0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]]

        action_magnitude:
            Size of the action (I did 3/4 of the lidar radius)

            Example: action_magnitude = lidar_radius*0.75

    Outputs:
        action:
            (x, y) goal coordinates where (0, 0) is the current location
            Tuple in shape of (1, 2)
    """
    # Set of actions
    actions = [(0, action_magnitude), (action_magnitude, action_magnitude), (action_magnitude, 0), (action_magnitude, -action_magnitude), 
                (0, -action_magnitude), (-action_magnitude, -action_magnitude), (-action_magnitude, 0), (-action_magnitude, action_magnitude)]

    # Load model
    model = torch.load('policy_net.pth')
    model.eval()

    # Get action probabilities
    action_probs = model.forward(lidar_reading)

    # Select highest action probability
    action_selection = torch.argmax(action_probs).item()

    # Get action
    action = actions[action_selection]

    return action
