"""
Pytorch network for deep q learning
"""

#######
# Setup
#######
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DeepQ(nn.Module):
    """
    Critic Network
    """
    def __init__(self, action_size):
        super(DeepQ, self).__init__()

        self.state_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 5, 1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 5, 1),
            nn.AdaptiveAvgPool2d((1, 8)),
            nn.Flatten()
        )
        self.network = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, state):
        """
        Foward pass on the network
        
        args:
            state (2d tensor): represents the state
            action (1d tensor): action taken
            
        returns:
            ntw_output (1x1 tensor): predicted reward
        """
        state_encoded = self.state_encoder(state)
        ntw_output = self.network(state_encoded)
        return ntw_output
    
class DeepQ_ST(nn.Module):
    """
    Critic Network
    """
    def __init__(self, action_size):
        super(DeepQ_ST, self).__init__()

        self.state_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 5, 1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 5, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.network = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, state, previous_action):
        """
        Foward pass on the network
        
        args:
            state (2d tensor): represents the state
            action (1d tensor): action taken
            
        returns:
            ntw_output (1x1 tensor): predicted reward
        """
        state_encoded = self.state_encoder(state)
        linear_input = torch.concatenate((state_encoded[0], previous_action[0]))
        ntw_output = self.network(linear_input)
        return ntw_output
    