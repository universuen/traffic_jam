import torch
from torch import nn


class PolicyModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, output_dim),
            nn.Softmax(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
