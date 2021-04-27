import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        
        
    def forward(self, x):
    	return self.fc(x)
