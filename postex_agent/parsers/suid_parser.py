"""Parser for SUID binary enumeration output."""
from __future__ import annotations

from pathlib import Path

from postex_agent.parsers.base_parser import BaseParser


_EXPLOITABLE_SUID_BINS = {
    "find",
    "vim",
    "nmap",
    "bash",
    "sh",
    "python",
    "python3",
    "perl",
    "cp",
    "tar",
}


class SuidParser(BaseParser):
    def parse(self, raw_output: str) -> dict:
        raw = raw_output or ""
        binaries = [line.strip() for line in raw.splitlines() if line.strip().startswith("/")]

        exploitable_bins: list[str] = []
        for binary in binaries:
            name = Path(binary).name.lower()
            if name in _EXPLOITABLE_SUID_BINS and binary not in exploitable_bins:
                exploitable_bins.append(binary)

        return {
            "vector_found": bool(exploitable_bins),
            "exploitable": bool(exploitable_bins),
            "details": {
                "raw": raw,
                "all_bins": binaries,
                "exploitable_bins": exploitable_bins,
            },
        }

