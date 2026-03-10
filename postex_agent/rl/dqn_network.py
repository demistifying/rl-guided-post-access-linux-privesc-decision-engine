"""
Dueling DQN network architecture (Wang et al., 2016).

Splits Q(s,a) into:
  V(s)   — state-value stream   (scalar)
  A(s,a) — advantage stream     (per-action)
  Q(s,a) = V(s) + A(s,a) - mean(A)
"""
from __future__ import annotations

import torch
from torch import nn


class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()

        # Shared feature layer
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)             # (batch, 1)
        advantage = self.advantage_stream(features)     # (batch, action_dim)
        # Q = V + (A - mean(A))  — mean subtraction for identifiability
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q
