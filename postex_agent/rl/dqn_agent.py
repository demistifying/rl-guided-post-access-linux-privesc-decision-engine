"""DQN agent: training, inference, save/load.

Supports:
- Action masking  (pass ``mask`` to ``select_action``)
- Prioritized experience replay  (PER)
- Configurable replay buffer size (default 50 000)
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from postex_agent.core.actions import ACTION_SPACE_SIZE, Action, compute_action_mask
from postex_agent.core.state import STATE_DIM
from postex_agent.rl.dqn_network import DQNNetwork
from postex_agent.rl.replay_buffer import PrioritizedReplayBuffer


@dataclass(frozen=True)
class DQNConfig:
    state_dim:               int   = STATE_DIM
    action_dim:              int   = ACTION_SPACE_SIZE
    hidden_dim:              int   = 64
    gamma:                   float = 0.95
    learning_rate:           float = 1e-3
    replay_buffer_size:      int   = 50_000
    batch_size:              int   = 64
    target_update_interval:  int   = 500
    epsilon_start:           float = 1.0
    epsilon_end:             float = 0.05
    epsilon_decay_episodes:  int   = 5_000
    # PER hyperparameters
    per_alpha:               float = 0.6
    per_beta_start:          float = 0.4
    per_beta_end:            float = 1.0
    per_beta_anneal_episodes: int  = 10_000


def build_action_mask_batch(state_batch: np.ndarray) -> np.ndarray:
    """Compute valid-action masks for a batch of state vectors."""
    if state_batch.ndim != 2 or state_batch.shape[1] != STATE_DIM:
        raise ValueError(
            f"Expected state batch with shape (N, {STATE_DIM}), got {state_batch.shape}"
        )

    masks = np.stack([compute_action_mask(state) for state in state_batch], axis=0)
    empty_rows = ~masks.any(axis=1)
    if np.any(empty_rows):
        masks[empty_rows, int(Action.STOP)] = True
    return masks.astype(np.bool_, copy=False)


def masked_double_dqn_next_q(
    online_q: torch.Tensor,
    target_q: torch.Tensor,
    action_masks: torch.Tensor,
) -> torch.Tensor:
    """Return masked Double DQN bootstrap values for a batch."""
    if online_q.shape != target_q.shape or online_q.shape != action_masks.shape:
        raise ValueError(
            "Expected online_q, target_q, and action_masks to share the same shape"
        )

    invalid_fill = torch.finfo(online_q.dtype).min
    masked_online = online_q.masked_fill(~action_masks, invalid_fill)
    next_actions = masked_online.argmax(dim=1, keepdim=True)
    masked_target = target_q.masked_fill(~action_masks, invalid_fill)
    return masked_target.gather(1, next_actions).squeeze(1)


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

        self.optimizer = Adam(self.online_net.parameters(), lr=self.config.learning_rate)

        # Prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.replay_buffer_size,
            alpha=self.config.per_alpha,
            seed=seed,
        )

        self.epsilon            = self.config.epsilon_start
        self.beta               = self.config.per_beta_start
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

    # ── Beta schedule (PER importance-sampling) ──────────────────────────

    def set_beta(self, episode_index: int) -> None:
        if episode_index >= self.config.per_beta_anneal_episodes:
            self.beta = self.config.per_beta_end
        else:
            frac = episode_index / float(self.config.per_beta_anneal_episodes)
            self.beta = self.config.per_beta_start + frac * (
                self.config.per_beta_end - self.config.per_beta_start
            )

    # ── Action selection ─────────────────────────────────────────────────

    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        mask: Optional[np.ndarray] = None,
    ) -> int:
        """Select an action, optionally constrained by *mask*.

        Parameters
        ----------
        state : (STATE_DIM,) float32 array
        explore : bool — use ε-greedy exploration
        mask : optional (ACTION_DIM,) bool array — True = valid, False = masked
        """
        if explore and self._rng.random() < self.epsilon:
            if mask is not None:
                valid = np.where(mask)[0]
                if len(valid) > 0:
                    return int(self._rng.choice(valid.tolist()))
            return self._rng.randrange(self.config.action_dim)

        t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(t).squeeze(0)

        if mask is not None:
            mask_t = torch.from_numpy(mask.astype(np.bool_)).to(self.device)
            q = q.masked_fill(~mask_t, float("-inf"))

        return int(torch.argmax(q).item())

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

        (states, actions, rewards, next_states, dones,
         tree_indices, is_weights) = self.replay_buffer.sample(
            self.config.batch_size, beta=self.beta
        )

        s  = torch.from_numpy(states).to(self.device)
        a  = torch.from_numpy(actions).to(self.device)
        r  = torch.from_numpy(rewards).to(self.device)
        ns = torch.from_numpy(next_states).to(self.device)
        d  = torch.from_numpy(dones).to(self.device)
        w  = torch.from_numpy(is_weights).to(self.device)

        q_vals = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        next_masks = torch.from_numpy(build_action_mask_batch(next_states)).to(self.device)
        with torch.no_grad():
            # Double DQN: online net selects action, target net evaluates it
            online_next_q = self.online_net(ns)
            target_next_q = self.target_net(ns)
            next_q = masked_double_dqn_next_q(
                online_next_q,
                target_next_q,
                next_masks,
            )
            targets = r + self.config.gamma * (1.0 - d) * next_q

        td_errors = (q_vals - targets).detach()

        # Weighted MSE loss (importance sampling)
        elementwise_loss = (q_vals - targets) ** 2
        loss = (w * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimization_steps += 1

        # Update priorities in the replay buffer
        self.replay_buffer.update_priorities(
            tree_indices, td_errors.cpu().numpy()
        )

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
