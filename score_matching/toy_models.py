import torch
import torch.nn as nn


class ToyScoreNetwork(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.fc1(x)
        x = self.softplus(x)
        x = self.fc2(x)
        x = self.softplus(x)
        x = self.fc3(x)
        return x
    

    