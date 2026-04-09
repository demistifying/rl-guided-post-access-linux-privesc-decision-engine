"""Parser for cron job enumeration output."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from postex_agent.parsers.base_parser import BaseParser

# Matches standard 5-field cron lines (including */N step notation)
# Also handles the optional "user" field in /etc/crontab system-wide format.
_CRON_LINE_RE = re.compile(
    r"^\s*(\*(?:/\d+)?|[\d,\-\/]+)\s+"   # minute
    r"(\*(?:/\d+)?|[\d,\-\/]+)\s+"        # hour
    r"(\*(?:/\d+)?|[\d,\-\/]+)\s+"        # day of month
    r"(\*(?:/\d+)?|[\d,\-\/]+)\s+"        # month
    r"(\*(?:/\d+)?|[\d,\-\/]+)\s+"        # day of week
    r"(.+)$"                               # [user] command
)

# @reboot / @hourly / @daily shorthand
_AT_LINE_RE = re.compile(r"^\s*(@\w+)\s+(.+)$")


class CronParser(BaseParser):
    def parse(self, raw_output: str) -> Dict[str, Any]:
        if not raw_output.strip():
            return {"vector_found": False, "details": {"cron_jobs": [], "potentially_writable": []}}

        cron_jobs: List[str]         = []
        writable_scripts: List[str]  = []

        for line in raw_output.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            command = ""
            m = _CRON_LINE_RE.match(stripped)
            if m:
                command = m.group(6).strip()
                cron_jobs.append(stripped)
            else:
                m2 = _AT_LINE_RE.match(stripped)
                if m2:
                    command = m2.group(2).strip()
                    cron_jobs.append(stripped)

            if command:
                # Flag scripts in writable dirs or with injectable extensions
                if any(d in command for d in ["/tmp/", "/var/tmp/", "/dev/shm/"]):
                    writable_scripts.append(command)
                if re.search(r"\.(sh|py|pl|rb|php)\b", command):
                    writable_scripts.append(command)

        writable_scripts = list(dict.fromkeys(writable_scripts))

        return {
            "vector_found": bool(cron_jobs),
            "details": {
                "cron_jobs":           cron_jobs,
                "writable_targets":    writable_scripts,
                "potentially_writable": writable_scripts,
            },
        }
