"""Action-to-command mapping with safety checks for live execution mode."""
from __future__ import annotations

import re
from typing import Dict, List

from postex_agent.core.actions import Action


_COMMANDS: Dict[Action, List[str]] = {
    Action.IDENTIFY_OS: [
        "uname -a",
        "cat /etc/os-release 2>/dev/null",
    ],
    Action.IDENTIFY_USER: [
        "id",
        "whoami",
    ],
    Action.CHECK_SUDO: [
        "sudo -l",
    ],
    Action.CHECK_SUID: [
        "find / -perm -4000 -type f 2>/dev/null",
    ],
    Action.CHECK_CAPABILITIES: [
        "getcap -r / 2>/dev/null",
    ],
    Action.CHECK_WRITABLE: [
        "find / -xdev -type d -perm -0002 2>/dev/null | head -n 200",
    ],
    Action.CHECK_CRON: [
        "cat /etc/crontab 2>/dev/null",
        "ls -la /etc/cron* 2>/dev/null",
    ],
    Action.SEARCH_CREDENTIALS: [
        "env | grep -iE 'pass|token|key|secret' 2>/dev/null",
        "grep -RniE 'password|passwd|token|secret|apikey' /etc /opt /home 2>/dev/null | head -n 200",
    ],
    Action.CHECK_KERNEL: [
        "uname -r",
    ],
    Action.EXPLOIT_SUDO: [],
    Action.EXPLOIT_SUID: [],
    Action.EXPLOIT_CAP: [],
    Action.EXPLOIT_CRON: [],
    Action.EXPLOIT_KERNEL: [],
    Action.VERIFY_ROOT: [
        "id",
        "whoami",
    ],
    Action.STOP: [],
}


_BLOCKLIST_PATTERNS: List[str] = [
    r"\brm\s+-rf\b",
    r"\bmkfs(?:\.\w+)?\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
    r"\bhalt\b",
    r"\binit\s+[06]\b",
    r"\bdd\s+if=",
    r"\bchmod\s+777\s+/\b",
    r">\s*/dev/sd[a-z]",
]

_BLOCKLIST_RE = [re.compile(pat, re.IGNORECASE) for pat in _BLOCKLIST_PATTERNS]


def get_commands(action: Action) -> List[str]:
    return list(_COMMANDS.get(action, []))


def is_safe_command(command: str) -> bool:
    cmd = command.strip()
    if not cmd:
        return False
    for pattern in _BLOCKLIST_RE:
        if pattern.search(cmd):
            return False
    return True

