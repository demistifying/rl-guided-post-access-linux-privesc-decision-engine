"""Abstract base class for all shell session adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSession(ABC):
    """
    All session adapters must implement `run`.

    Returns a dict:
        {
            "output":    str  - combined stdout
            "error":     str  - stderr or error message
            "exit_code": int  - 0 on success, non-zero on failure
        }
    """

    @abstractmethod
    def run(self, command: str, timeout: int = 30) -> dict:
        """Execute a shell command and return structured output."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources."""
        ...
