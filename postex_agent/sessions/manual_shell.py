"""
Manual shell session adapter.

On POSIX targets this adapter keeps a persistent shell process alive so command
state such as cwd and exported environment variables survives between calls.
On Windows it falls back to one-command-per-process execution, which is still
useful for local testing in this repository.
"""
from __future__ import annotations

import os
import select
import subprocess
import threading
import time
import uuid
from typing import Optional

from postex_agent.sessions.base_session import BaseSession


class ManualShellSession(BaseSession):
    """Run commands against a local shell for testing or on-target execution."""

    def __init__(self, shell: Optional[str] = None, persistent: Optional[bool] = None):
        if shell is None:
            shell = "powershell" if os.name == "nt" else "/bin/sh"
        self.shell = shell
        self._persistent_requested = persistent if persistent is not None else (os.name != "nt")
        self._persistent_supported = self._persistent_requested and (os.name != "nt")
        self._process: Optional[subprocess.Popen[str]] = None
        self._io_lock = threading.Lock()

    def metadata(self) -> dict:
        return {
            "session_type": self.__class__.__name__,
            "persistent_context": self._persistent_supported,
            "shell": self.shell,
        }

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process

        self._process = subprocess.Popen(
            [self.shell],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        return self._process

    def _reset_process(self) -> None:
        if self._process is None:
            return
        try:
            if self._process.stdin:
                self._process.stdin.close()
        except OSError:
            pass
        try:
            self._process.terminate()
            self._process.wait(timeout=1)
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass
        self._process = None

    def _run_persistent_posix(self, command: str, timeout: int) -> dict:
        proc = self._ensure_process()
        assert proc.stdin is not None
        assert proc.stdout is not None

        sentinel = f"__CODEX_SESSION_{uuid.uuid4().hex}__"
        wrapped = (
            f"{command}\n"
            "status=$?\n"
            f"printf '{sentinel}:%s\\n' \"$status\"\n"
        )

        with self._io_lock:
            try:
                proc.stdin.write(wrapped)
                proc.stdin.flush()
            except Exception as exc:
                self._reset_process()
                return {
                    "output": "",
                    "error": f"Failed to write to persistent shell: {exc}",
                    "exit_code": 1,
                }

            output_lines = []
            deadline = time.time() + timeout

            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    self._reset_process()
                    return {
                        "output": "".join(output_lines),
                        "error": f"Command timed out after {timeout}s",
                        "exit_code": 124,
                    }

                ready, _, _ = select.select([proc.stdout], [], [], remaining)
                if not ready:
                    continue

                line = proc.stdout.readline()
                if line == "":
                    if proc.poll() is None:
                        continue
                    self._reset_process()
                    return {
                        "output": "".join(output_lines),
                        "error": "Persistent shell exited unexpectedly.",
                        "exit_code": 1,
                    }

                stripped = line.rstrip("\r\n")
                if stripped.startswith(f"{sentinel}:"):
                    exit_text = stripped.split(":", 1)[1].strip()
                    try:
                        exit_code = int(exit_text)
                    except ValueError:
                        exit_code = 1
                    return {
                        "output": "".join(output_lines),
                        "error": "",
                        "exit_code": exit_code,
                    }

                output_lines.append(line)

    def _run_stateless(self, command: str, timeout: int) -> dict:
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
                "output": completed.stdout or "",
                "error": completed.stderr or "",
                "exit_code": int(completed.returncode),
            }
        except subprocess.TimeoutExpired:
            return {
                "output": "",
                "error": f"Command timed out after {timeout}s",
                "exit_code": 124,
            }
        except Exception as exc:
            return {
                "output": "",
                "error": str(exc),
                "exit_code": 1,
            }

    def run(self, command: str, timeout: int = 30) -> dict:
        if self._persistent_supported:
            return self._run_persistent_posix(command, timeout=timeout)
        return self._run_stateless(command, timeout=timeout)

    def close(self) -> None:
        self._reset_process()
