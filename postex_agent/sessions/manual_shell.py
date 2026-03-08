"""
Manual shell session adapter.

Executes commands via subprocess on the local machine.

Use this when:
  - The agent is running directly on the target (e.g. uploaded via web shell)
  - You are doing local testing

For remote targets use SSHSession or MetasploitSession instead.
"""
from __future__ import annotations

import os
import subprocess

from postex_agent.sessions.base_session import BaseSession


class ManualShellSession(BaseSession):
    """
    Runs each command in a fresh subprocess shell.
    Returns structured dict matching the BaseSession contract.
    """

    def __init__(self, shell: str | None = None):
        if shell is None:
            shell = "powershell" if os.name == "nt" else "/bin/sh"
        self.shell = shell

    def run(self, command: str, timeout: int = 30) -> dict:
        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                executable=self.shell if os.name != "nt" else None,
            )
            return {
                "output":    completed.stdout or "",
                "error":     completed.stderr or "",
                "exit_code": int(completed.returncode),
            }
        except subprocess.TimeoutExpired:
            return {
                "output":    "",
                "error":     f"Command timed out after {timeout}s",
                "exit_code": 124,
            }
        except Exception as exc:
            return {
                "output":    "",
                "error":     str(exc),
                "exit_code": 1,
            }

    def close(self) -> None:
        pass
