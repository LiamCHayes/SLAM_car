"""
High-level planner

Gets action from the neural net

Put the model in the same directory as this script
"""

import torch

class ActionSelection:
    """
    Inputs:
        action_magnitude:
            Size of the action (I did 3/4 of the lidar radius)

            Example: action_magnitude = lidar_radius*0.75
    """
    def __init__(self, action_magnitude):
        self.action_magnitude = action_magnitude
        self.prev_action_selection = None
        self.actions = [(0, action_magnitude), (action_magnitude, action_magnitude), (action_magnitude, 0), (action_magnitude, -action_magnitude), 
                    (0, -action_magnitude), (-action_magnitude, -action_magnitude), (-action_magnitude, 0), (-action_magnitude, action_magnitude)]

    def get_action(self, lidar_reading):
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

        Outputs:
            action:
                (x, y) goal coordinates where (0, 0) is the current location
                Tuple in shape of (1, 2)
        """
        # Load model
        model = torch.load('policy_net.pth')
        model.eval()

        # Get action probabilities
        action_probs = model.forward(lidar_reading)
        if self.prev_act_selection is not None:
            zero_action = self.prev_act_selection + 4 if self.prev_act_selection < 4 else self.prev_act_selection - 4
            action_probs[0, zero_action] = 0

        # Get action
        action_selection = torch.argmax(action_probs).item()
        action = self.actions[action_selection]

        return action
