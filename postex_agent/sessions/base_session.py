"""Abstract command session interface."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSession(ABC):
    @abstractmethod
    def run(self, command: str, timeout: int = 30) -> dict:
        """Execute command and return {'output', 'error', 'exit_code'}."""

    @abstractmethod
    def close(self) -> None:
        """Close underlying resources."""

