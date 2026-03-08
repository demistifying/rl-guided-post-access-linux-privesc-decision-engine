"""Base parser for command output parsing."""
from __future__ import annotations


class BaseParser:
    def parse(self, raw_output: str) -> dict:
        return {
            "vector_found": False,
            "exploitable": False,
            "details": {"raw": raw_output or ""},
        }

