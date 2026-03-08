"""Parser for `sudo -l` output."""
from __future__ import annotations

import re
from pathlib import Path

from postex_agent.parsers.base_parser import BaseParser


_NOPASSWD_RE = re.compile(r"NOPASSWD:\s*(.+)$", re.IGNORECASE)
_SUDO_ENTRY_RE = re.compile(r"^\s*\(.*\)\s+(.+)$")

_GTFOBINS_HINTS = {
    "find",
    "vim",
    "nmap",
    "less",
    "awk",
    "perl",
    "python",
    "python3",
    "bash",
    "sh",
    "tar",
}


class SudoParser(BaseParser):
    def parse(self, raw_output: str) -> dict:
        raw = raw_output or ""
        lines = raw.splitlines()

        sudo_commands: list[str] = []
        nopasswd_entries: list[str] = []
        exploitable_bins: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            sudo_match = _SUDO_ENTRY_RE.match(stripped)
            if sudo_match:
                command = sudo_match.group(1).strip()
                sudo_commands.append(command)

            np_match = _NOPASSWD_RE.search(stripped)
            if np_match:
                entry = np_match.group(1).strip()
                nopasswd_entries.append(entry)
                for token in re.split(r"[,\s]+", entry):
                    token = token.strip()
                    if token.startswith("/"):
                        base = Path(token).name.lower()
                        if base in _GTFOBINS_HINTS and token not in exploitable_bins:
                            exploitable_bins.append(token)

        denied = "not allowed to execute sudo" in raw.lower() or "may not run sudo" in raw.lower()
        vector_found = bool(nopasswd_entries or exploitable_bins) and not denied

        return {
            "vector_found": vector_found,
            "exploitable": vector_found,
            "details": {
                "raw": raw,
                "sudo_commands": sudo_commands,
                "nopasswd_entries": nopasswd_entries,
                "exploitable_bins": exploitable_bins,
            },
        }

