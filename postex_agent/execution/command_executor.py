"""Safety-checked command execution with structured JSONL logging."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

from postex_agent.environment.command_library import is_safe_command
from postex_agent.sessions.base_session import BaseSession


class CommandExecutor:
    def __init__(self, session: BaseSession, log_path: str = "logs/execution.jsonl"):
        self.session = session
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def _write_log(self, payload: Dict[str, Any]) -> None:
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def execute(self, command: str, action_name: str = "", timeout: int = 30) -> Dict[str, Any]:
        started = time.time()
        blocked = not is_safe_command(command)

        if blocked:
            result = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "action": action_name,
                "command": command,
                "blocked": True,
                "output": "",
                "error": "Command blocked by safety policy",
                "exit_code": 126,
                "duration_ms": int((time.time() - started) * 1000),
            }
            self._write_log(result)
            return result

        execution_result = self.session.run(command, timeout=timeout)
        result = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action_name,
            "command": command,
            "blocked": False,
            "output": execution_result.get("output", ""),
            "error": execution_result.get("error", ""),
            "exit_code": int(execution_result.get("exit_code", 1)),
            "duration_ms": int((time.time() - started) * 1000),
        }
        self._write_log(result)
        return result

