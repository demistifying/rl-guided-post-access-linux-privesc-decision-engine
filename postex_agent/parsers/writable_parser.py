"""Parser for writable path enumeration output."""
from __future__ import annotations

from postex_agent.parsers.base_parser import BaseParser


class WritableParser(BaseParser):
    def parse(self, raw_output: str) -> dict:
        raw = raw_output or ""
        writable_paths = [line.strip() for line in raw.splitlines() if line.strip().startswith("/")]
        return {
            "vector_found": bool(writable_paths),
            "exploitable": bool(writable_paths),
            "details": {
                "raw": raw,
                "writable_paths": writable_paths,
            },
        }

