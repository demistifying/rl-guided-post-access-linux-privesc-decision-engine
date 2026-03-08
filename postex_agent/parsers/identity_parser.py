"""Parsers for OS and user identification outputs."""
from __future__ import annotations

import re
from typing import Optional

from postex_agent.parsers.base_parser import BaseParser


_KERNEL_RE = re.compile(r"\b(\d+\.\d+\.\d+(?:[-._a-zA-Z0-9]+)?)\b")
_PRETTY_NAME_RE = re.compile(r'^PRETTY_NAME="?([^"\n]+)"?', re.MULTILINE)
_UID_RE = re.compile(r"uid=(\d+)\(([^)]+)\)")


class OSParser(BaseParser):
    def parse(self, raw_output: str) -> dict:
        raw = raw_output or ""
        kernel_match = _KERNEL_RE.search(raw)
        pretty_match = _PRETTY_NAME_RE.search(raw)

        kernel_version: Optional[str] = kernel_match.group(1) if kernel_match else None
        os_name: Optional[str] = pretty_match.group(1).strip() if pretty_match else None
        if os_name is None and "Linux" in raw:
            os_name = "Linux"

        return {
            "vector_found": bool(kernel_version or os_name),
            "exploitable": False,
            "details": {
                "raw": raw,
                "kernel_version": kernel_version,
                "os_name": os_name,
            },
        }


class UserParser(BaseParser):
    def parse(self, raw_output: str) -> dict:
        raw = raw_output or ""
        uid_match = _UID_RE.search(raw)

        uid = int(uid_match.group(1)) if uid_match else None
        username = uid_match.group(2) if uid_match else None

        if username is None:
            whoami = raw.strip().splitlines()
            if len(whoami) == 1 and whoami[0]:
                username = whoami[0].strip()
                uid = 0 if username == "root" else uid

        is_root = bool(uid == 0 or username == "root" or "uid=0(" in raw)

        return {
            "vector_found": bool(username is not None or uid is not None),
            "exploitable": is_root,
            "details": {
                "raw": raw,
                "uid": uid,
                "username": username,
                "is_root": is_root,
            },
        }

