import torch
import numpy as np
import torch.nn as nn


class hybridModel(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(hybridModel, self).__init__()
        self.baseLayer1 = nn.LSTM(input_dim, 16, 1, 
                                  batch_first=True, 
                                  bidirectional=False)
        self.baseLayer2 = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(16, 64),
            nn.ReLU(),
        )
        self.adaptiveLayer = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        x, _ = self.baseLayer1(x)
        x = self.baseLayer2(x)
        x = self.adaptiveLayer(x)
        return x
    

class transferLearningBench(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(transferLearningBench, self).__init__()
        self.baseLayer1 = nn.LSTM(input_dim, 16, 1, 
                                  batch_first=True, 
                                  bidirectional=False)
        self.baseLayer2 = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(16, 64),
            nn.ReLU(),
        )
        self.adaptiveLayer = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.tanh(x)
        x, _ = self.baseLayer1(x)
        x = self.baseLayer2(x)
        x = self.adaptiveLayer(x)
        return x