"""
Replay buffers for DQN training.

Provides:
  - ReplayBuffer:            uniform random sampling (original)
  - PrioritizedReplayBuffer: proportional priority sampling via SumTree
"""
from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


# ── Uniform replay buffer (kept for backward-compat) ─────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        transition: Transition = (
            state.astype(np.float32, copy=True),
            int(action),
            float(reward),
            next_state.astype(np.float32, copy=True),
            bool(done),
        )
        self._buffer.append(transition)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch: List[Transition] = self._rng.sample(self._buffer, batch_size)
        states = np.stack([item[0] for item in batch]).astype(np.float32)
        actions = np.array([item[1] for item in batch], dtype=np.int64)
        rewards = np.array([item[2] for item in batch], dtype=np.float32)
        next_states = np.stack([item[3] for item in batch]).astype(np.float32)
        dones = np.array([item[4] for item in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones


# ── SumTree for O(log n) proportional sampling ───────────────────────────

class SumTree:
    """Binary tree where each leaf holds a priority value.

    Parent nodes store the sum of children, enabling O(log n)
    proportional sampling and priority update.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data: List[Transition | None] = [None] * capacity
        self._write_idx = 0
        self._size = 0

    @property
    def total(self) -> float:
        return float(self._tree[0])

    def __len__(self) -> int:
        return self._size

    def add(self, priority: float, data: Transition) -> None:
        tree_idx = self._write_idx + self.capacity - 1
        self._data[self._write_idx] = data
        self._update(tree_idx, priority)
        self._write_idx = (self._write_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _update(self, tree_idx: int, priority: float) -> None:
        change = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self._tree[tree_idx] += change

    def update(self, tree_idx: int, priority: float) -> None:
        self._update(tree_idx, priority)

    def get(self, cumsum: float) -> Tuple[int, float, Transition]:
        """Walk down the tree to find the leaf for *cumsum*."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self._tree):
                break
            if cumsum <= self._tree[left]:
                idx = left
            else:
                cumsum -= self._tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self._tree[idx], self._data[data_idx]  # type: ignore[return-value]


# ── Prioritized experience replay buffer ─────────────────────────────────

class PrioritizedReplayBuffer:
    """Proportional prioritized replay (Schaul et al., 2015).

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    alpha : float
        Exponent controlling how much prioritization is used (0 = uniform).
    seed : int
        Random seed for reproducibility.
    """

    PER_EPSILON = 1e-5          # small constant added to TD error
    MAX_PRIORITY_INIT = 1.0     # initial max priority for new transitions

    def __init__(self, capacity: int, alpha: float = 0.6, seed: int = 0):
        self.capacity = capacity
        self.alpha = alpha
        self._tree = SumTree(capacity)
        self._max_priority = self.MAX_PRIORITY_INIT
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._tree)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        transition: Transition = (
            state.astype(np.float32, copy=True),
            int(action),
            float(reward),
            next_state.astype(np.float32, copy=True),
            bool(done),
        )
        priority = self._max_priority ** self.alpha
        self._tree.add(priority, transition)

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """Sample a batch with importance-sampling weights.

        Returns
        -------
        states, actions, rewards, next_states, dones, tree_indices, is_weights
        """
        n = len(self._tree)
        total = self._tree.total

        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        batch: List[Transition] = []

        segment = total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            cumsum = self._rng.uniform(low, high)
            tree_idx, priority, data = self._tree.get(cumsum)
            indices[i] = tree_idx
            priorities[i] = priority
            batch.append(data)

        # Importance sampling weights
        probs = priorities / total
        weights = (n * probs) ** (-beta)
        weights /= weights.max()  # normalize

        states = np.stack([item[0] for item in batch]).astype(np.float32)
        actions = np.array([item[1] for item in batch], dtype=np.int64)
        rewards = np.array([item[2] for item in batch], dtype=np.float32)
        next_states = np.stack([item[3] for item in batch]).astype(np.float32)
        dones = np.array([item[4] for item in batch], dtype=np.float32)

        return (
            states, actions, rewards, next_states, dones,
            indices, weights.astype(np.float32),
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities for sampled transitions after learning."""
        priorities = (np.abs(td_errors) + self.PER_EPSILON) ** self.alpha
        for idx, prio in zip(tree_indices, priorities):
            self._tree.update(int(idx), float(prio))
            self._max_priority = max(self._max_priority, float(prio) ** (1.0 / self.alpha))
