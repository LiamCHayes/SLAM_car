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
    def __init__(self, action_size, episode_length):
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
        self.lstm = nn.LSTM(episode_length, 40, 2, batch_first=True)
        self.lstm_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80, 8), 
            nn.ReLU()
        )
        self.network = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, state, path):
        """
        Foward pass on the network
        
        args:
            state (2d tensor): represents the state
            action (1d tensor): action taken
            
        returns:
            ntw_output (1x1 tensor): predicted reward
        """
        # Encode state
        state_encoded = self.state_encoder(state) 

        # Encode path
        sequences = []
        for p_idx in range(path.shape[0]):
            p = path[p_idx, :, :]
            pad_size = 25 - p.shape[1]
            p_padded = torch.nn.functional.pad(p, (0, pad_size))
            sequences.append(p_padded)
        path_padded = torch.stack(sequences, dim=0)
        output, _ = self.lstm(path_padded)
        seq_encoded = self.lstm_encoder(output)

        # Put together
        linear_input = torch.cat((state_encoded, seq_encoded), dim=1)
        ntw_output = self.network(linear_input)

        return ntw_output
    