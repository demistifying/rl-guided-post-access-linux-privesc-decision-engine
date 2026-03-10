"""
Policy inference module.
Loads a trained DQN checkpoint and exposes `predict(state_vector) -> Action`.
Used by the real environment CLI during live pentesting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from postex_agent.core.actions import Action, ACTION_DESCRIPTIONS, compute_action_mask
from postex_agent.rl.dqn_agent import DQNAgent, DQNConfig


class RLPolicy:
    """
    Wraps a trained DQN agent for inference-only use.
    Thread-safe for single-threaded CLI use.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}\n"
                "Train first with: python -m postex_agent.rl.train_dqn"
            )
        self._agent = DQNAgent(config=DQNConfig(), seed=0, device=device)
        self._agent.load(model_path)
        self._model_path = model_path

    def predict(self, state_vector: np.ndarray) -> Action:
        """Return the greedy action for the given state."""
        mask = compute_action_mask(state_vector)
        action_id = self._agent.select_action(state_vector, explore=False, mask=mask)
        return Action(action_id)

    def q_values(self, state_vector: np.ndarray) -> dict:
        """Return Q-values per action (for CLI display)."""
        qv = self._agent.q_values(state_vector)
        mask = compute_action_mask(state_vector)
        return {Action(i): float(qv[i]) for i in range(len(qv)) if mask[i]}

    def top_actions(self, state_vector: np.ndarray, n: int = 3) -> list:
        """Return top-n ranked valid actions by Q-value."""
        qv = self._agent.q_values(state_vector)
        mask = compute_action_mask(state_vector)
        ranked = sorted(
            [(i, v) for i, v in enumerate(qv) if mask[i]],
            key=lambda x: x[1], reverse=True,
        )
        return [(Action(i), float(v), ACTION_DESCRIPTIONS.get(Action(i), "")) for i, v in ranked[:n]]

    @property
    def model_path(self) -> str:
        return self._model_path
