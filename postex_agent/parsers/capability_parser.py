"""Parser for Linux capability enumeration output."""
from __future__ import annotations

from postex_agent.parsers.base_parser import BaseParser


_DANGEROUS_CAPS = ("cap_setuid", "cap_setgid")


class CapabilityParser(BaseParser):
    def parse(self, raw_output: str) -> dict:
        raw = raw_output or ""
        exploitable_binaries: list[str] = []

        for line in raw.splitlines():
            stripped = line.strip()
            if "=" not in stripped:
                continue
            binary, caps = [part.strip() for part in stripped.split("=", 1)]
            if any(cap in caps for cap in _DANGEROUS_CAPS):
                exploitable_binaries.append(binary)

        return {
            "vector_found": bool(exploitable_binaries),
            "exploitable": bool(exploitable_binaries),
            "details": {
                "raw": raw,
                "exploitable_binaries": exploitable_binaries,
            },
        }

