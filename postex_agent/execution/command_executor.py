"""Safety-checked command execution with structured JSONL logging."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from postex_agent.environment.command_library import is_safe_command
from postex_agent.sessions.base_session import BaseSession


class CommandExecutor:
    """
    Executes whitelisted commands on a BaseSession.
    All session adapters return {'output', 'error', 'exit_code'} dicts.
    Every execution is appended to a JSONL log file.
    """

    def __init__(self, session: BaseSession, log_path: str = "logs/execution.jsonl"):
        self.session  = session
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def execute(self, command: str, action_name: str = "", timeout: int = 30) -> Dict[str, Any]:
        """
        Safety-check then execute one command.

        Returns:
            {
                "ts":          str   - ISO timestamp
                "action":      str   - action name for context
                "command":     str   - the command string
                "blocked":     bool  - True if safety policy blocked it
                "output":      str   - stdout from target
                "error":       str   - stderr / error message
                "exit_code":   int   - process exit code
                "duration_ms": int   - wall-clock milliseconds
            }
        """
        started = time.time()
        session_meta = self.session.metadata()

        if not is_safe_command(command):
            entry = {
                "ts":          datetime.now(timezone.utc).isoformat(),
                "action":      action_name,
                "session_type": session_meta.get("session_type", ""),
                "persistent_context": bool(session_meta.get("persistent_context", False)),
                "command":     command,
                "blocked":     True,
                "output":      "",
                "error":       "Command blocked by safety policy.",
                "exit_code":   126,
                "duration_ms": 0,
            }
            self._write_log(entry)
            return entry

        session_result = self.session.run(command, timeout=timeout)
        entry = {
            "ts":          datetime.now(timezone.utc).isoformat(),
            "action":      action_name,
            "session_type": session_meta.get("session_type", ""),
            "persistent_context": bool(session_meta.get("persistent_context", False)),
            "command":     command,
            "blocked":     False,
            "output":      session_result.get("output", ""),
            "error":       session_result.get("error", ""),
            "exit_code":   int(session_result.get("exit_code", 0)),
            "duration_ms": int((time.time() - started) * 1000),
        }
        self._write_log(entry)
        return entry

    def execute_all(self, commands: List[str], action_name: str = "") -> List[Dict[str, Any]]:
        return [self.execute(cmd, action_name=action_name) for cmd in commands]

    def _write_log(self, entry: Dict[str, Any]) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass
