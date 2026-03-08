"""Parser for cron enumeration output."""
from __future__ import annotations

from postex_agent.parsers.base_parser import BaseParser


class CronParser(BaseParser):
    def parse(self, raw_output: str) -> dict:
        raw = raw_output or ""
        cron_jobs: list[str] = []

        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # heuristic: cron lines usually have at least 6 columns
            if len(stripped.split()) >= 6:
                cron_jobs.append(stripped)

        return {
            "vector_found": bool(cron_jobs),
            "exploitable": bool(cron_jobs),
            "details": {
                "raw": raw,
                "cron_jobs": cron_jobs,
            },
        }

