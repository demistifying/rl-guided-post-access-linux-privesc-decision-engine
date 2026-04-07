"""Parser for basic credential discovery patterns with quality classification.

Credential quality levels (matching simulation_env):
  0.33 — hash (bcrypt, sha512, md5, etc.)
  0.66 — plaintext password / token
  1.00 — private key or root password
"""
from __future__ import annotations

import re

from postex_agent.parsers.base_parser import BaseParser


_KEYWORDS = ("password", "passwd", "token", "secret", "api_key", "apikey", "api-key")

# Hash patterns — common Linux crypt and web-app hash formats
_HASH_PATTERNS = [
    re.compile(r"\$[0-9a-z]\$"),            # $6$, $5$, $1$, $2b$, $y$
    re.compile(r"^[a-f0-9]{32}$"),           # MD5 hex
    re.compile(r"^[a-f0-9]{40}$"),           # SHA-1 hex
    re.compile(r"^[a-f0-9]{64}$"),           # SHA-256 hex
]

# Private key markers
_KEY_MARKERS = (
    "BEGIN RSA PRIVATE KEY",
    "BEGIN DSA PRIVATE KEY",
    "BEGIN EC PRIVATE KEY",
    "BEGIN OPENSSH PRIVATE KEY",
    "BEGIN PRIVATE KEY",
)


def _classify_credential(value: str) -> float:
    """Return quality score for a single credential value."""
    v = value.strip()

    # Private key → highest quality
    for marker in _KEY_MARKERS:
        if marker in v:
            return 1.0

    # Hash → lowest quality (needs cracking)
    for pattern in _HASH_PATTERNS:
        if pattern.search(v):
            return 0.33

    # Plaintext password / token → medium quality
    return 0.66


class CredentialParser(BaseParser):
    def parse(self, raw_output: str) -> dict:
        raw = raw_output or ""
        credentials: list[str] = []
        qualities: list[float] = []

        # Check for private keys in the full output first
        has_key = any(marker in raw for marker in _KEY_MARKERS)

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
                qualities.append(_classify_credential(value))

        # If a private key was found anywhere in the output, add it
        if has_key and not any(q >= 1.0 for q in qualities):
            credentials.append("[SSH/RSA private key detected]")
            qualities.append(1.0)

        best_quality = max(qualities) if qualities else 0.0

        return {
            "vector_found": bool(credentials),
            "exploitable": bool(credentials),
            "details": {
                "raw": raw,
                "credentials": credentials,
                "cred_count": len(credentials),
                "cred_quality": best_quality,
            },
        }
