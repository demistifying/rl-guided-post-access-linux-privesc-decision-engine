from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from actions import Action
from dqn_network import DQNNetwork
from replay_buffer import ReplayBuffer
from state import STATE_DIM


@dataclass(frozen=True)
class DQNConfig:
    state_dim: int = STATE_DIM
    action_dim: int = len(Action)
    hidden_dim: int = 64
    gamma: float = 0.95
    learning_rate: float = 1e-3
    replay_buffer_size: int = 10_000
    batch_size: int = 64
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 5_000


class DQNAgent:
    def __init__(
        self,
        config: Optional[DQNConfig] = None,
        seed: int = 0,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or DQNConfig()
        self._rng = random.Random(seed)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.online_network = DQNNetwork(
            input_dim=self.config.state_dim,
            output_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)
        self.target_network = DQNNetwork(
            input_dim=self.config.state_dim,
            output_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = Adam(self.online_network.parameters(), lr=self.config.learning_rate)
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size, seed=seed)

        self.epsilon = self.config.epsilon_start
        self.optimization_steps = 0

    def set_epsilon_for_episode(self, episode_index: int) -> None:
        if episode_index >= self.config.epsilon_decay_episodes:
            self.epsilon = self.config.epsilon_end
            return
        fraction = episode_index / float(self.config.epsilon_decay_episodes)
        self.epsilon = self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.config.action_dim)

        state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        q_values = self.online_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states_t).max(dim=1).values
            targets = rewards_t + self.config.gamma * (1.0 - dones_t) * next_q_values

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimization_steps += 1
        return float(loss.item())

    def hard_update_target(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.online_network.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.online_network.load_state_dict(state_dict)
        self.hard_update_target()
