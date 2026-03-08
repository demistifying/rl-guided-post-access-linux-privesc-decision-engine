"""DQN agent: training, inference, save/load."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from postex_agent.core.actions import ACTION_SPACE_SIZE
from postex_agent.core.state import STATE_DIM
from postex_agent.rl.dqn_network import DQNNetwork
from postex_agent.rl.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class DQNConfig:
    state_dim:               int   = STATE_DIM
    action_dim:              int   = ACTION_SPACE_SIZE
    hidden_dim:              int   = 64
    gamma:                   float = 0.95
    learning_rate:           float = 1e-3
    replay_buffer_size:      int   = 10_000
    batch_size:              int   = 64
    target_update_interval:  int   = 500
    epsilon_start:           float = 1.0
    epsilon_end:             float = 0.05
    epsilon_decay_episodes:  int   = 5_000


class DQNAgent:
    def __init__(
        self,
        config:  Optional[DQNConfig] = None,
        seed:    int = 0,
        device:  Optional[str] = None,
    ) -> None:
        self.config = config or DQNConfig()
        self._rng   = random.Random(seed)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.online_net = DQNNetwork(
            input_dim=self.config.state_dim,
            output_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)
        self.target_net = DQNNetwork(
            input_dim=self.config.state_dim,
            output_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer     = Adam(self.online_net.parameters(), lr=self.config.learning_rate)
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size, seed=seed)

        self.epsilon            = self.config.epsilon_start
        self.optimization_steps = 0

    # ── Epsilon schedule ─────────────────────────────────────────────────

    def set_epsilon(self, episode_index: int) -> None:
        if episode_index >= self.config.epsilon_decay_episodes:
            self.epsilon = self.config.epsilon_end
        else:
            frac = episode_index / float(self.config.epsilon_decay_episodes)
            self.epsilon = self.config.epsilon_start + frac * (
                self.config.epsilon_end - self.config.epsilon_start
            )

    # ── Action selection ─────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.config.action_dim)
        t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(t)
        return int(torch.argmax(q, dim=1).item())

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return full Q-value vector (useful for CLI display)."""
        t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(t)
        return q.cpu().numpy().squeeze()

    # ── Learning ─────────────────────────────────────────────────────────

    def observe(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )
        s  = torch.from_numpy(states).to(self.device)
        a  = torch.from_numpy(actions).to(self.device)
        r  = torch.from_numpy(rewards).to(self.device)
        ns = torch.from_numpy(next_states).to(self.device)
        d  = torch.from_numpy(dones).to(self.device)

        q_vals    = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q    = self.target_net(ns).max(dim=1).values
            targets   = r + self.config.gamma * (1.0 - d) * next_q

        loss = F.mse_loss(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimization_steps += 1
        return float(loss.item())

    def hard_update_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "online_net":           self.online_net.state_dict(),
            "config":               self.config.__dict__,
            "optimization_steps":   self.optimization_steps,
        }, path)

    def load(self, path: str) -> None:
        # PyTorch >=2.6 defaults to weights_only=True, which blocks custom dataclasses.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        state_dict = checkpoint
        optimization_steps = 0

        if isinstance(checkpoint, dict):
            if "online_net" in checkpoint and isinstance(checkpoint["online_net"], dict):
                # New checkpoint format saved by this project.
                state_dict = checkpoint["online_net"]
                optimization_steps = int(checkpoint.get("optimization_steps", 0))
            elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                # Common compatibility format used by some wrappers.
                state_dict = checkpoint["state_dict"]
                optimization_steps = int(checkpoint.get("optimization_steps", 0))

        self.online_net.load_state_dict(state_dict)
        self.hard_update_target()
        self.optimization_steps = optimization_steps
