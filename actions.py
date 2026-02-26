from __future__ import annotations

from enum import IntEnum
from typing import Dict, List


class Action(IntEnum):
    SYSTEM_IDENTIFICATION = 0
    USER_IDENTIFICATION = 1
    CHECK_SUDO = 2
    CHECK_SUID = 3
    CHECK_CAP = 4
    CHECK_WRITABLE = 5
    CHECK_CRON = 6
    CHECK_CREDENTIALS = 7
    CHECK_KERNEL = 8
    ESCALATE = 9
    STOP = 10


VECTOR_KEYS: List[str] = [
    "sudo",
    "suid",
    "capabilities",
    "writable_path",
    "cron",
    "credentials",
    "kernel",
]


CHECK_ACTIONS: List[Action] = [
    Action.CHECK_SUDO,
    Action.CHECK_SUID,
    Action.CHECK_CAP,
    Action.CHECK_WRITABLE,
    Action.CHECK_CRON,
    Action.CHECK_CREDENTIALS,
    Action.CHECK_KERNEL,
]


VECTOR_BY_CHECK_ACTION: Dict[Action, str] = {
    Action.CHECK_SUDO: "sudo",
    Action.CHECK_SUID: "suid",
    Action.CHECK_CAP: "capabilities",
    Action.CHECK_WRITABLE: "writable_path",
    Action.CHECK_CRON: "cron",
    Action.CHECK_CREDENTIALS: "credentials",
    Action.CHECK_KERNEL: "kernel",
}


CHECK_ACTION_BY_VECTOR: Dict[str, Action] = {
    vector: action for action, vector in VECTOR_BY_CHECK_ACTION.items()
}

