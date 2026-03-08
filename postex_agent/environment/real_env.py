"""Live command execution environment wrapper for post-access decision support."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from postex_agent.core.actions import Action
from postex_agent.core.state import HostState
from postex_agent.environment.command_library import get_commands
from postex_agent.environment.parser_registry import parse_output
from postex_agent.environment.state_builder import update_state
from postex_agent.execution.command_executor import CommandExecutor
from postex_agent.sessions.base_session import BaseSession


StepResult = Tuple[np.ndarray, float, bool, Dict[str, Any]]


class RealEnv:
    """
    Minimal gym-style wrapper over a real session.
    Reward shaping is not used in live mode; returns 0.0 reward and state transitions.
    """

    def __init__(self, session: BaseSession, max_steps: int = 30, log_path: str = "logs/execution.jsonl"):
        self.executor = CommandExecutor(session=session, log_path=log_path)
        self.max_steps = max_steps
        self.state = HostState()
        self.steps = 0
        self.done = False

    def reset(self) -> np.ndarray:
        self.state = HostState()
        self.steps = 0
        self.done = False
        return self.state.to_vector()

    def step(self, action: int | Action) -> StepResult:
        if self.done:
            raise RuntimeError("Environment is done. Call reset().")
        action_enum = Action(int(action))
        commands = get_commands(action_enum)

        output = ""
        exec_results = []
        for command in commands:
            result = self.executor.execute(command=command, action_name=action_enum.name)
            exec_results.append(result)
            output += (result.get("output") or "") + "\n"

        parsed = parse_output(action_enum, output)
        update_state(self.state, action_enum, parsed)

        self.steps += 1
        if action_enum == Action.STOP or self.state.current_privilege == 1 or self.steps >= self.max_steps:
            self.done = True

        info: Dict[str, Any] = {
            "action": action_enum.name,
            "commands": commands,
            "execution": exec_results,
            "parsed": parsed,
        }
        return self.state.to_vector(), 0.0, self.done, info

