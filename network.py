import torch
from torch import nn
import numpy as np



class FeedForwardNN(nn.Module):
    def __init__(self, in_dim=18, out_dim=1, act_min=-1, act_max=1):
        super(FeedForwardNN, self).__init__()

        # Lower and upper limits of each action
        self.act_min = act_min
        if not isinstance(self.act_min, torch.Tensor):
            self.act_min = torch.tensor(self.act_min)

        self.act_max = act_max
        if not isinstance(self.act_max, torch.Tensor):
            self.act_max = torch.tensor(self.act_max)

        # Layers of the net, numbers are arbitrary
        self.features = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        output = self.features(obs)
        output = torch.clamp(input=output, min=self.act_min, max=self.act_max)
        return output.float()
