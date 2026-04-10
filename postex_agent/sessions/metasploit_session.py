"""
Optional session adapters for Metasploit RPC and SSH.

MetasploitSession requires:  pip install pymetasploit3
SSHSession requires:         pip install paramiko
"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Optional

from postex_agent.sessions.base_session import BaseSession


def _build_msf_wrapped_command(command: str) -> tuple[str, str, str]:
    """Wrap a shell command so msfrpc output includes an exit code and terminator."""
    token = uuid.uuid4().hex
    status_token = f"__CODEX_MSF_STATUS_{token}__"
    done_token = f"__CODEX_MSF_DONE_{token}__"
    wrapped = (
        f"{command}\n"
        "status=$?\n"
        f"printf '{status_token}:%s\\n' \"$status\"\n"
        f"printf '{done_token}\\n'\n"
    )
    return wrapped, status_token, done_token


def _parse_msf_wrapped_output(raw_output: str, status_token: str, done_token: str) -> dict:
    """Strip wrapper markers from msfrpc output and recover the shell exit code."""
    normalized = (raw_output or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    clean_lines = []
    exit_code: Optional[int] = None
    saw_done = False

    for line in lines:
        if line.startswith(f"{status_token}:"):
            exit_text = line.split(":", 1)[1].strip()
            try:
                exit_code = int(exit_text)
            except ValueError:
                exit_code = 1
            continue
        if line.strip().startswith(done_token):
            saw_done = True
            continue
        clean_lines.append(line)

    output = "\n".join(clean_lines).rstrip("\n")
    if not saw_done:
        return {
            "output": output,
            "error": "Metasploit session output was incomplete; completion sentinel not found.",
            "exit_code": 1,
        }
    return {
        "output": output,
        "error": "",
        "exit_code": int(exit_code if exit_code is not None else 0),
    }


class MetasploitSession(BaseSession):
    """
    Executes commands through an active Metasploit session via msfrpc.

    Commands are wrapped with explicit completion and exit-code markers so the
    caller gets more reliable output than a simple newline-terminated read.
    """

    def __init__(
        self,
        session_id: int,
        host: str = "127.0.0.1",
        port: int = 55553,
        password: str = "msf",
        ssl: bool = False,
        client_factory: Optional[Callable[..., Any]] = None,
        session: Any = None,
        reconnect_attempts: int = 1,
    ):
        self.session_id = int(session_id)
        self.host = host
        self.port = int(port)
        self.password = password
        self.ssl = bool(ssl)
        self._client_factory = client_factory
        self._client = None
        self._session = None
        self._reconnect_attempts = max(int(reconnect_attempts), 0)

        if session is not None:
            self._session = session
        else:
            self._connect()

    def _make_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory(
                password=self.password,
                server=self.host,
                port=self.port,
                ssl=self.ssl,
            )

        try:
            from pymetasploit3.msfrpc import MsfRpcClient  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pymetasploit3 is required for MetasploitSession.\n"
                "Install with: pip install pymetasploit3"
            ) from exc

        return MsfRpcClient(
            password=self.password,
            server=self.host,
            port=self.port,
            ssl=self.ssl,
        )

    def _connect(self) -> None:
        self._client = self._make_client()
        self._session = self._client.sessions.session(str(self.session_id))

    def _reset_connection(self) -> None:
        self._session = None
        self._client = None

    def _ensure_session(self) -> None:
        if self._session is None:
            self._connect()

    def _run_once(self, command: str, timeout: int) -> dict:
        self._ensure_session()
        wrapped, status_token, done_token = _build_msf_wrapped_command(command)
        
        if type(self._session).__name__ == "MeterpreterSession":
            try:
                # Meterpreter needs a specific method to drop into an OS shell
                raw_output = self._session.run_shell_cmd_with_output(
                    wrapped,
                    end_strs=[done_token],
                )
            except TypeError:
                raw_output = self._session.run_shell_cmd_with_output(wrapped)
        else:
            try:
                raw_output = self._session.run_with_output(
                    wrapped,
                    terminating_strs=[done_token],
                    timeout=timeout,
                )
            except TypeError:
                raw_output = self._session.run_with_output(wrapped, timeout=timeout)
            
        return _parse_msf_wrapped_output(raw_output or "", status_token, done_token)

    def run(self, command: str, timeout: int = 30) -> dict:
        last_error: Optional[Exception] = None

        for attempt in range(self._reconnect_attempts + 1):
            try:
                return self._run_once(command, timeout=timeout)
            except Exception as exc:
                last_error = exc
                self._reset_connection()
                if attempt >= self._reconnect_attempts:
                    break

        return {"output": "", "error": str(last_error) if last_error else "Unknown error", "exit_code": 1}

    def metadata(self) -> dict:
        return {
            "session_type": self.__class__.__name__,
            "persistent_context": True,
            "host": self.host,
            "port": self.port,
            "session_id": self.session_id,
            "ssl": self.ssl,
        }

    def close(self) -> None:
        self._reset_connection()


class SSHSession(BaseSession):
    """SSH session adapter using paramiko."""

    def __init__(
        self,
        host: str,
        username: str,
        password: str | None = None,
        key_path: str | None = None,
        port: int = 22,
        timeout: int = 10,
    ):
        try:
            import paramiko  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "paramiko is required for SSHSession.\n"
                "Install with: pip install paramiko"
            ) from exc

        self.host = host
        self.username = username
        self.port = int(port)

        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs: dict[str, Any] = {
            "hostname": host,
            "username": username,
            "port": port,
            "timeout": timeout,
        }
        if key_path:
            connect_kwargs["key_filename"] = key_path
        elif password:
            connect_kwargs["password"] = password
        self._client.connect(**connect_kwargs)

    def run(self, command: str, timeout: int = 30) -> dict:
        try:
            _, stdout, stderr = self._client.exec_command(command, timeout=timeout)
            output = stdout.read().decode(errors="replace")
            error = stderr.read().decode(errors="replace")
            exit_code = stdout.channel.recv_exit_status()
            return {"output": output, "error": error, "exit_code": int(exit_code)}
        except Exception as exc:
            return {"output": "", "error": str(exc), "exit_code": 1}

    def metadata(self) -> dict:
        return {
            "session_type": self.__class__.__name__,
            "persistent_context": False,
            "host": self.host,
            "port": self.port,
            "username": self.username,
        }

    def close(self) -> None:
        if self._client:
            self._client.close()
