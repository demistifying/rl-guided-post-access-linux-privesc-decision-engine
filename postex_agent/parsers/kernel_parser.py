"""Parser for kernel version detection and simple known-CVE matching."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from postex_agent.parsers.base_parser import BaseParser


_KERNEL_VERSION_RE = re.compile(r"\b(\d+\.\d+\.\d+(?:[-._a-zA-Z0-9]+)?)\b")


def _parse_version_tuple(version: str) -> Optional[Tuple[int, int, int]]:
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


@dataclass(frozen=True)
class KnownKernelIssue:
    cve: str
    name: str
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]


_KNOWN_ISSUES: List[KnownKernelIssue] = [
    # Dirty Pipe (approx range)
    KnownKernelIssue("CVE-2022-0847", "Dirty Pipe", (5, 8, 0), (5, 19, 0)),
    # Dirty COW (broad historical range)
    KnownKernelIssue("CVE-2016-5195", "Dirty COW", (2, 6, 22), (4, 8, 3)),
]


class KernelParser(BaseParser):
    def parse(self, raw_output: str) -> dict:
        raw = raw_output or ""
        match = _KERNEL_VERSION_RE.search(raw)
        kernel_version = match.group(1) if match else None

        known_cves: list[dict] = []
        parsed = _parse_version_tuple(kernel_version) if kernel_version else None
        if parsed is not None:
            for issue in _KNOWN_ISSUES:
                if issue.lower <= parsed <= issue.upper:
                    known_cves.append({"cve": issue.cve, "name": issue.name})

        return {
            "vector_found": bool(known_cves),
            "exploitable": bool(known_cves),
            "details": {
                "raw": raw,
                "kernel_version": kernel_version,
                "known_cves": known_cves,
            },
        }

