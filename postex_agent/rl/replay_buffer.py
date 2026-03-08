from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


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

