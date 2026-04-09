"""Shared live-session execution controller for CLI and RealEnv."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from postex_agent.core.actions import Action, VECTOR_BY_EXPLOIT_ACTION
from postex_agent.core.state import HostState
from postex_agent.environment.command_library import get_commands
from postex_agent.environment.parser_registry import parse_output
from postex_agent.environment.state_builder import (
    VECTOR_RISK_PENALTIES,
    update_state,
    update_temporal,
)
from postex_agent.execution.command_executor import CommandExecutor
from postex_agent.sessions.base_session import BaseSession


LiveStepResult = Tuple[np.ndarray, float, bool, Dict[str, Any]]


class LiveExecutionController:
    """Owns live state transitions for real shell-backed operation."""

    def __init__(
        self,
        session: BaseSession,
        max_steps: int = 30,
        log_path: str = "logs/execution.jsonl",
    ):
        self.executor = CommandExecutor(session=session, log_path=log_path)
        self.max_steps = max_steps
        self.state = HostState()
        self.steps = 0
        self.done = False
        self._cumulative_risk = 0.0

    def reset(self) -> np.ndarray:
        self.state = HostState()
        self.steps = 0
        self.done = False
        self._cumulative_risk = 0.0
        return self.state.to_vector()

    def preview_commands(self, action: int | Action) -> list[str]:
        action_enum = Action(int(action))
        return get_commands(action_enum, state=self.state)

    def step(self, action: int | Action, timeout: int = 30) -> LiveStepResult:
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        action_enum = Action(int(action))
        commands = self.preview_commands(action_enum)
        state_before = self.state.to_dict()

        combined_output = ""
        exec_results = []

        for command in commands:
            if command.strip().startswith("#"):
                continue

            result = self.executor.execute(
                command=command,
                action_name=action_enum.name,
                timeout=timeout,
            )
            exec_results.append(result)
            combined_output += result.get("output", "") + "\n"

        parsed = parse_output(action_enum, combined_output)
        update_state(self.state, action_enum, parsed)

        if action_enum in VECTOR_BY_EXPLOIT_ACTION:
            vector = VECTOR_BY_EXPLOIT_ACTION[action_enum]
            self._cumulative_risk += VECTOR_RISK_PENALTIES.get(vector, 0.0)

            # ── Post-exploit auto-verification ────────────────────────
            # Real exploits typically produce no output (they spawn a
            # shell), so the parser can't detect success.  Run `id` to
            # check whether we actually escalated.
            if self.state.current_privilege != 1:
                verify_result = self.executor.execute(
                    command="id",
                    action_name="auto_verify",
                    timeout=5,
                )
                exec_results.append(verify_result)
                verify_parsed = parse_output(
                    Action.VERIFY_ROOT,
                    verify_result.get("output", ""),
                )
                update_state(self.state, Action.VERIFY_ROOT, verify_parsed)

        self.steps += 1
        update_temporal(self.state, self.steps, self.max_steps, self._cumulative_risk)

        if (
            action_enum == Action.STOP
            or self.state.current_privilege == 1
            or self.steps >= self.max_steps
        ):
            self.done = True

        info: Dict[str, Any] = {
            "action": action_enum.name,
            "commands": commands,
            "execution": exec_results,
            "parsed": parsed,
            "state_before": state_before,
            "state_after": self.state.to_dict(),
            "step": self.steps,
        }
        return self.state.to_vector(), 0.0, self.done, info

    @property
    def current_state(self) -> HostState:
        return self.state
