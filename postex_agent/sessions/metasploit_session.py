"""
Optional session adapters for Metasploit RPC and SSH.

MetasploitSession requires:  pip install pymetasploit3
SSHSession requires:         pip install paramiko
"""
from __future__ import annotations

from postex_agent.sessions.base_session import BaseSession


class MetasploitSession(BaseSession):
    """
    Executes commands through an active Metasploit session via msfrpc.
    """

    def __init__(
        self,
        session_id: int,
        host:       str = "127.0.0.1",
        port:       int = 55553,
        password:   str = "msf",
        ssl:        bool = False,
    ):
        try:
            from pymetasploit3.msfrpc import MsfRpcClient  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pymetasploit3 is required for MetasploitSession.\n"
                "Install with: pip install pymetasploit3"
            ) from exc

        self._client  = MsfRpcClient(password=password, server=host, port=port, ssl=ssl)
        self._session = self._client.sessions.session(str(session_id))

    def run(self, command: str, timeout: int = 30) -> dict:
        try:
            output = self._session.run_with_output(
                command, terminating_strs=["\n"], timeout=timeout,
            )
            return {"output": output or "", "error": "", "exit_code": 0}
        except Exception as exc:
            return {"output": "", "error": str(exc), "exit_code": 1}

    def close(self) -> None:
        self._session = None
        self._client  = None


class SSHSession(BaseSession):
    """
    SSH session adapter using paramiko.
    """

    def __init__(
        self,
        host:      str,
        username:  str,
        password:  str | None = None,
        key_path:  str | None = None,
        port:      int = 22,
        timeout:   int = 10,
    ):
        try:
            import paramiko  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "paramiko is required for SSHSession.\n"
                "Install with: pip install paramiko"
            ) from exc

        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs: dict = {
            "hostname": host, "username": username,
            "port": port, "timeout": timeout,
        }
        if key_path:
            connect_kwargs["key_filename"] = key_path
        elif password:
            connect_kwargs["password"] = password
        self._client.connect(**connect_kwargs)

    def run(self, command: str, timeout: int = 30) -> dict:
        try:
            _, stdout, stderr = self._client.exec_command(command, timeout=timeout)
            out       = stdout.read().decode(errors="replace")
            err       = stderr.read().decode(errors="replace")
            exit_code = stdout.channel.recv_exit_status()
            return {"output": out, "error": err, "exit_code": int(exit_code)}
        except Exception as exc:
            return {"output": "", "error": str(exc), "exit_code": 1}

    def close(self) -> None:
        if self._client:
            self._client.close()
