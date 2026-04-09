"""Parser for SUID binary enumeration output."""
from __future__ import annotations

import re
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
    "env",
    "ruby",
    "node",
    "php",
    "strace",
    "dash",
}


def _normalise_bin_name(path: str) -> str:
    """Strip version suffixes: python3.10 → python3, vim.basic → vim."""
    name = Path(path).name.lower()
    name = re.sub(r"\.basic$", "", name)       # vim.basic → vim
    name = re.sub(r"\.\d+$", "", name)         # python3.10 → python3
    name = re.sub(r"(\d)\.\d+$", r"\1", name)  # perl5.34 → perl5
    return name


class SuidParser(BaseParser):
    def parse(self, raw_output: str) -> dict:
        raw = raw_output or ""
        binaries = [line.strip() for line in raw.splitlines() if line.strip().startswith("/")]

        exploitable_bins: list[str] = []
        for binary in binaries:
            name = _normalise_bin_name(binary)
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

