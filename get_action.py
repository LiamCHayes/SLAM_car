"""
High-level planner

Gets action from the neural net

Put the model in the same directory as this script
"""

import torch
import numpy as np
import math

# Class to do a forward pass on the NN
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
        Returns the x, y goal coordinates to go to for the low level planner

        Inputs:
            lidar_reading:
                Output of self.transform_lidar()
                An nxn numpy array where n is the lidar radius
                0 is a free space and 1 is an obstacle or unknown

        Outputs:
            action:
                (x, y) goal coordinates where (0, 0) is the current location
                Tuple in shape of (1, 2)
        """
        # Load model
        model = torch.load('policy_net.pth')
        model.eval()

        # Forward pass
        lidar_reading = self._np_to_tensor(lidar_reading).unsqueeze(0)
        action_probs = model.forward(lidar_reading)

        # Make sure we don't choose the opposite action of the one before
        if self.prev_act_selection is not None:
            zero_action = self.prev_act_selection + 4 if self.prev_act_selection < 4 else self.prev_act_selection - 4
            action_probs[0, zero_action] = 0

        # Get action
        action_selection = torch.argmax(action_probs).item()
        action = self.actions[action_selection]

        return action
    
    def transform_lidar(self, angle_min, angle_increment, ranges):
        """
        Transforms the lidar reading from ROS to input for the NN

        Inputs:
            angle_min: from the ros reading
            angle_increment: from the ros reading
            ranges: from the ros reading

        Returns:
            lidar_reading: numpy arrray lidar reading ready to input into self.get_action()
        """
        # Get polar coordinates and cut off distances within range
        thetas = []
        rs = []
        for i, r in enumerate(ranges):
            # Get r in a format that works with the second loop
            if math.isnan(r):
                r = np.nan
            else:
                r = round(r*100) if r < 8. and r > 0.5 else np.nan

            # Append the values
            rs.append(r)
            thetas.append(angle_min + i * angle_increment)

        # Convert form polar readings to numpy array
        npoints = len(thetas)
        lidar_reading = np.zeros((801, 801))
        np_origin = 400
        r_matrix, theta_matrix = self._get_polar_matrices(np_origin)
        for i in range(npoints):
            r = rs[i]
            theta = thetas[i]
            prev_theta_idx = i-1 if i > 0 else npoints-1
            next_theta_idx = i + 1 if i < npoints-1 else 0
            if not np.isnan(r):
                # Find all thetas that are most similar to this theta
                rows = []
                cols = []
                for row in range(theta_matrix.shape[0]):
                    for col in range(theta_matrix.shape[1]):
                        dist_i_minus1 = np.abs(thetas[prev_theta_idx] - theta_matrix[row, col])
                        dist_i = np.abs(theta - theta_matrix[row, col])
                        dist_i_plus1 = np.abs(thetas[next_theta_idx] - theta_matrix[row, col])
                        if dist_i < dist_i_minus1 and dist_i < dist_i_plus1:
                            rows.append(row)
                            cols.append(col)

                # If we can't see it, set the value to 1
                for p in range(len(rows)):
                    point_r = r_matrix[rows[p], cols[p]]
                    if point_r >= r:
                        lidar_reading[rows[p], cols[p]] = 1

        return lidar_reading

    def _get_polar_matrices(self, np_origin):
        """
        Utility function for transforming the lidar reading
        """
        r_matrix = np.zeros((np_origin*2+1, np_origin*2+1))
        theta_matrix = np.zeros((np_origin*2+1, np_origin*2+1))
        for row in range(r_matrix.shape[0]):
            for col in range(r_matrix.shape[1]):
                x = col - np_origin
                y = -(row - np_origin)
                r_matrix[row, col] = round((x**2 + y**2) ** 0.5)
                if x != 0:
                    if x > 0 and y > 0:
                        theta_matrix[row, col] = np.arctan(y/x)
                    elif x < 0:
                        theta_matrix[row, col] = np.arctan(y/x) + np.pi
                    elif x > 0 and y < 0:
                        theta_matrix[row, col] = np.arctan(y/x) + np.pi * 2
                else:
                    if y > 0:
                        theta_matrix[row, col] = np.pi/2
                    elif y < 0:
                        theta_matrix[row, col] = 3*np.pi/2
        
        return r_matrix, theta_matrix
    
    def _np_to_tensor(self, arr: np.ndarray):
        """
        Utility function to convert numpy array to the right format of tensor
        """
        arr = arr.astype(np.float32)
        tensor = torch.from_numpy(arr)
        tensor = tensor.unsqueeze(0)
        return tensor
