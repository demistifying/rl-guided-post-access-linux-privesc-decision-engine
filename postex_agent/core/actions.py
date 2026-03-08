"""
Action space definition shared between RL training and real deployment.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Dict, List


class Action(IntEnum):
    IDENTIFY_OS         = 0
    IDENTIFY_USER       = 1
    CHECK_SUDO          = 2
    CHECK_SUID          = 3
    CHECK_CAPABILITIES  = 4
    CHECK_WRITABLE      = 5
    CHECK_CRON          = 6
    SEARCH_CREDENTIALS  = 7
    CHECK_KERNEL        = 8
    EXPLOIT_SUDO        = 9
    EXPLOIT_SUID        = 10
    EXPLOIT_CAP         = 11
    EXPLOIT_CRON        = 12
    EXPLOIT_KERNEL      = 13
    VERIFY_ROOT         = 14
    STOP                = 15


ACTION_SPACE_SIZE: int = len(Action)

# Maps check actions → vector key names (matching state.VECTOR_KEYS)
VECTOR_BY_CHECK_ACTION: Dict[Action, str] = {
    Action.CHECK_SUDO:         "sudo",
    Action.CHECK_SUID:         "suid",
    Action.CHECK_CAPABILITIES: "capabilities",
    Action.CHECK_WRITABLE:     "writable_path",
    Action.CHECK_CRON:         "cron",
    Action.SEARCH_CREDENTIALS: "credentials",
    Action.CHECK_KERNEL:       "kernel",
}

# Maps exploit actions → vector key names
VECTOR_BY_EXPLOIT_ACTION: Dict[Action, str] = {
    Action.EXPLOIT_SUDO:   "sudo",
    Action.EXPLOIT_SUID:   "suid",
    Action.EXPLOIT_CAP:    "capabilities",
    Action.EXPLOIT_CRON:   "cron",
    Action.EXPLOIT_KERNEL: "kernel",
}

CHECK_ACTIONS: List[Action] = list(VECTOR_BY_CHECK_ACTION.keys())
EXPLOIT_ACTIONS: List[Action] = list(VECTOR_BY_EXPLOIT_ACTION.keys())

IDENTIFICATION_ACTIONS: List[Action] = [
    Action.IDENTIFY_OS,
    Action.IDENTIFY_USER,
]

ACTION_DESCRIPTIONS: Dict[Action, str] = {
    Action.IDENTIFY_OS:        "Identify operating system and kernel version",
    Action.IDENTIFY_USER:      "Identify current user and privileges",
    Action.CHECK_SUDO:         "Check sudo permissions (sudo -l)",
    Action.CHECK_SUID:         "Search for SUID binaries",
    Action.CHECK_CAPABILITIES: "Check Linux capabilities",
    Action.CHECK_WRITABLE:     "Find world-writable files and paths",
    Action.CHECK_CRON:         "Enumerate cron jobs and scheduled tasks",
    Action.SEARCH_CREDENTIALS: "Search for credentials in files/env",
    Action.CHECK_KERNEL:       "Check kernel version for known exploits",
    Action.EXPLOIT_SUDO:       "Exploit sudo misconfiguration",
    Action.EXPLOIT_SUID:       "Exploit SUID binary",
    Action.EXPLOIT_CAP:        "Exploit Linux capabilities",
    Action.EXPLOIT_CRON:       "Exploit writable cron job",
    Action.EXPLOIT_KERNEL:     "Exploit kernel vulnerability",
    Action.VERIFY_ROOT:        "Verify root access (id, whoami)",
    Action.STOP:               "Stop execution",
}
