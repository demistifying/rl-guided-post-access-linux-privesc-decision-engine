"""Optional adapters for Metasploit RPC and SSH sessions."""
from __future__ import annotations

from postex_agent.sessions.base_session import BaseSession


class MetasploitSession(BaseSession):
    def __init__(self, session_id: int, host: str = "127.0.0.1", port: int = 55553, password: str = "msf"):
        try:
            from pymetasploit3.msfrpc import MsfRpcClient  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pymetasploit3 is required for MetasploitSession.") from exc

        self._client = MsfRpcClient(password=password, server=host, port=port, ssl=False)
        self._session = self._client.sessions.session(str(session_id))

    def run(self, command: str, timeout: int = 30) -> dict:
        try:
            output = self._session.run_with_output(command, terminating_strs=["\n"], timeout=timeout)
            return {"output": output or "", "error": "", "exit_code": 0}
        except Exception as exc:  # pragma: no cover - network dependent
            return {"output": "", "error": str(exc), "exit_code": 1}

    def close(self) -> None:
        return None


class SSHSession(BaseSession):
    def __init__(
        self,
        host: str,
        username: str,
        password: str | None = None,
        key_path: str | None = None,
        port: int = 22,
    ):
        try:
            import paramiko  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("paramiko is required for SSHSession.") from exc

        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        kwargs: dict = {"hostname": host, "username": username, "port": port, "timeout": 10}
        if key_path:
            kwargs["key_filename"] = key_path
        else:
            kwargs["password"] = password
        self._client.connect(**kwargs)

    def run(self, command: str, timeout: int = 30) -> dict:
        try:
            stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)
            out = stdout.read().decode(errors="replace")
            err = stderr.read().decode(errors="replace")
            exit_code = stdout.channel.recv_exit_status()
            return {"output": out, "error": err, "exit_code": int(exit_code)}
        except Exception as exc:  # pragma: no cover - network dependent
            return {"output": "", "error": str(exc), "exit_code": 1}

    def close(self) -> None:
        self._client.close()

