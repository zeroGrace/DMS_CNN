import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_layers, out_dim):
        super().__init__()
        
        self.layers = nn.ModuleList() 
        
        self.layers.append(nn.Linear(in_dim, hidden_layers[0]))
        self.layers.append(nn.ReLU())
        
        for i in range(len(hidden_layers)-1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.ReLU())
            
        self.layers.append(nn.Linear(hidden_layers[-1], out_dim))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x