"""Live command execution environment wrapper for post-access decision support."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from postex_agent.core.actions import Action
from postex_agent.core.state import HostState
from postex_agent.execution.live_runtime import LiveExecutionController
from postex_agent.sessions.base_session import BaseSession


StepResult = Tuple[np.ndarray, float, bool, Dict[str, Any]]


class RealEnv:
    """
    Wrap a live shell session with the same ``step()`` interface as ``SimulationEnv``.

    Live mode does not shape reward today, so ``step()`` always returns ``0.0``.
    The shared live runtime owns execution, parsing, and state updates.
    """

    def __init__(
        self,
        session: BaseSession,
        max_steps: int = 30,
        log_path: str = "logs/execution.jsonl",
    ):
        self.runtime = LiveExecutionController(
            session=session,
            max_steps=max_steps,
            log_path=log_path,
        )

    def reset(self) -> np.ndarray:
        return self.runtime.reset()

    def step(self, action: int | Action) -> StepResult:
        return self.runtime.step(action)

    @property
    def current_state(self) -> HostState:
        return self.runtime.current_state
