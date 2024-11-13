"""
Pytorch network
"""

#######
# Setup
#######
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#######
# Model
#######
class Actor(nn.Module):
    """
    Actor network
    
    args:
        lidar_radius (int)
        
    attributes:
        lidar_radius (int)
        state_encoder (torch.nn.Sequential): encodes the state
        means (torch.nn.Sequential): returns the mean for a 2d multivariate normal
        cov_matrix (torch.nn.Sequential): returns a flattened covariance matrix for the 2d multivariate normal
    """
    def __init__(self, lidar_radius):
        super(Actor, self).__init__()

        self.lidar_radius = lidar_radius
        self.state_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 5, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 8)),
            nn.Flatten()
        )
        self.means = nn.Sequential(
            nn.Linear(128, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        self.cov_matrix = nn.Sequential(
            nn.Linear(128, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, 3)
        )
    
    def forward(self, x):
        """
        Forward pass and rescale by lidar radius

        args:
            x (tensor): input for the network
        
        returns:
            means (2x1 tensor)
            cov_matrix (2x2 tensor)
        """
        # Encode state
        encoded_state = self.state_encoder(x)

        # Get mean and cov matrix from encoded state
        means = self.means(encoded_state)
        cov_matrix = self.cov_matrix(encoded_state)

        # ensure a positive definite covariance matrix
        batch_size = cov_matrix.size(0)
        L = torch.zeros(batch_size, 2, 2).to(device)
        tril_indices = torch.tril_indices(row=2, col=2, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = cov_matrix
        L.diagonal(dim1=-2, dim2=-1).exp_()
        S = torch.matmul(L, L.transpose(-1, -2))

        return means, S
    
    def sample(self, x):
        """
        Samples an action from the output distribution
        
        args:
            x (tensor): Input state
            noise (bool): whether to add noise or not
        """
        means, cov_matrix = self.forward(x)
        mvn = MultivariateNormal(means, cov_matrix)
        action = mvn.sample()
        log_prob = mvn.log_prob(action).sum(axis=-1, keepdim=True)

        return action, log_prob
    
    def sample_pink(self, x, noise):
        """
        Samples an action from the output distribution
        
        args:
            x (tensor): Input state
            noise (bool): whether to add noise or not
        """
        means, cov_matrix = self.forward(x)
        mvn = MultivariateNormal(means, cov_matrix)
        action = mvn.sample() + noise
        log_prob = mvn.log_prob(action).sum(axis=-1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    """
    Critic Network
    """
    def __init__(self):
        super(Critic, self).__init__()

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
            nn.Linear(66, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        """
        Foward pass on the network
        
        args:
            state (2d tensor): represents the state
            action (1d tensor): action taken
            
        returns:
            ntw_output (1x1 tensor): predicted reward
        """
        state_encoded = self.state_encoder(state)
        x = torch.cat((state_encoded, action), dim=1)
        ntw_output = self.network(x)
        return ntw_output
