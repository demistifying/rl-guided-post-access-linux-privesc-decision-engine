"""Parser for basic credential discovery patterns."""
from __future__ import annotations

from postex_agent.parsers.base_parser import BaseParser


_KEYWORDS = ("password", "passwd", "token", "secret", "api_key", "apikey", "api-key")


class CredentialParser(BaseParser):
    def parse(self, raw_output: str) -> dict:
        raw = raw_output or ""
        credentials: list[str] = []

        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "=" not in stripped and ":" not in stripped:
                continue

            sep = "=" if "=" in stripped else ":"
            key, value = stripped.split(sep, 1)
            key_l = key.lower().strip()
            if not any(token in key_l for token in _KEYWORDS):
                continue
            entry = f"{key.strip()}{sep}{value.strip()}"
            if entry not in credentials:
                credentials.append(entry)

        return {
            "vector_found": bool(credentials),
            "exploitable": bool(credentials),
            "details": {
                "raw": raw,
                "credentials": credentials,
            },
        }
