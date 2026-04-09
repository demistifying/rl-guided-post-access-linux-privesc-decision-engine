"""Action space definition shared between RL training and real deployment."""
from __future__ import annotations

from enum import IntEnum
from typing import Dict, List

import numpy as np

from postex_agent.core.state import (
    CHECK_START_IDX,
    CURRENT_PRIVILEGE_IDX,
    EXPLOIT_FAIL_START_IDX,
    FOUND_START_IDX,
    MAX_EXPLOIT_FAILURES,
    OS_IDENTIFIED_IDX,
    STATE_DIM,
    USER_IDENTIFIED_IDX,
    VECTOR_KEYS,
)


class Action(IntEnum):
    IDENTIFY_OS = 0
    IDENTIFY_USER = 1
    CHECK_SUDO = 2
    CHECK_SUID = 3
    CHECK_CAPABILITIES = 4
    CHECK_WRITABLE = 5
    CHECK_CRON = 6
    SEARCH_CREDENTIALS = 7
    CHECK_KERNEL = 8
    EXPLOIT_SUDO = 9
    EXPLOIT_SUID = 10
    EXPLOIT_CAP = 11
    EXPLOIT_CRON = 12
    EXPLOIT_KERNEL = 13
    EXPLOIT_CREDENTIALS = 14
    EXPLOIT_WRITABLE = 15
    VERIFY_ROOT = 16
    STOP = 17


ACTION_SPACE_SIZE: int = len(Action)

VECTOR_BY_CHECK_ACTION: Dict[Action, str] = {
    Action.CHECK_SUDO: "sudo",
    Action.CHECK_SUID: "suid",
    Action.CHECK_CAPABILITIES: "capabilities",
    Action.CHECK_WRITABLE: "writable_path",
    Action.CHECK_CRON: "cron",
    Action.SEARCH_CREDENTIALS: "credentials",
    Action.CHECK_KERNEL: "kernel",
}

VECTOR_BY_EXPLOIT_ACTION: Dict[Action, str] = {
    Action.EXPLOIT_SUDO: "sudo",
    Action.EXPLOIT_SUID: "suid",
    Action.EXPLOIT_CAP: "capabilities",
    Action.EXPLOIT_CRON: "cron",
    Action.EXPLOIT_KERNEL: "kernel",
    Action.EXPLOIT_CREDENTIALS: "credentials",
    Action.EXPLOIT_WRITABLE: "writable_path",
}

CHECK_ACTIONS: List[Action] = list(VECTOR_BY_CHECK_ACTION.keys())
EXPLOIT_ACTIONS: List[Action] = list(VECTOR_BY_EXPLOIT_ACTION.keys())

IDENTIFICATION_ACTIONS: List[Action] = [
    Action.IDENTIFY_OS,
    Action.IDENTIFY_USER,
]

ACTION_DESCRIPTIONS: Dict[Action, str] = {
    Action.IDENTIFY_OS: "Identify operating system and kernel version",
    Action.IDENTIFY_USER: "Identify current user and privileges",
    Action.CHECK_SUDO: "Check sudo permissions (sudo -l)",
    Action.CHECK_SUID: "Search for SUID binaries",
    Action.CHECK_CAPABILITIES: "Check Linux capabilities",
    Action.CHECK_WRITABLE: "Find world-writable files and paths",
    Action.CHECK_CRON: "Enumerate cron jobs and scheduled tasks",
    Action.SEARCH_CREDENTIALS: "Search for credentials in files/env",
    Action.CHECK_KERNEL: "Check kernel version for known exploits",
    Action.EXPLOIT_SUDO: "Exploit sudo misconfiguration",
    Action.EXPLOIT_SUID: "Exploit SUID binary",
    Action.EXPLOIT_CAP: "Exploit Linux capabilities",
    Action.EXPLOIT_CRON: "Exploit writable cron job",
    Action.EXPLOIT_KERNEL: "Exploit kernel vulnerability",
    Action.EXPLOIT_CREDENTIALS: "Escalate using discovered credentials (su/ssh)",
    Action.EXPLOIT_WRITABLE: "Exploit writable PATH directory for hijacking",
    Action.VERIFY_ROOT: "Verify root access (id, whoami)",
    Action.STOP: "Stop execution",
}

EXPLOIT_RETRY_BUDGETS: Dict[str, int] = {
    "sudo": 1,
    "suid": 1,
    "capabilities": 1,
    "cron": 1,
    "kernel": 3,
    "credentials": 2,
    "writable_path": 1,
}


def exploit_retry_budget(vector: str) -> int:
    return EXPLOIT_RETRY_BUDGETS.get(vector, 1)


_CHECK_ACTION_VEC_IDX: Dict[int, int] = {
    int(action): VECTOR_KEYS.index(vector)
    for action, vector in VECTOR_BY_CHECK_ACTION.items()
}

_EXPLOIT_ACTION_VEC_IDX: Dict[int, int] = {
    int(action): VECTOR_KEYS.index(vector)
    for action, vector in VECTOR_BY_EXPLOIT_ACTION.items()
}


def compute_action_mask(state_vector: np.ndarray) -> np.ndarray:
    """Return a boolean mask of shape ``(ACTION_SPACE_SIZE,)``."""
    if state_vector.shape[0] != STATE_DIM:
        raise ValueError(f"Expected state vector with dim {STATE_DIM}, got {state_vector.shape[0]}")

    mask = np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)

    if state_vector[OS_IDENTIFIED_IDX] >= 0.5:
        mask[int(Action.IDENTIFY_OS)] = False

    if state_vector[USER_IDENTIFIED_IDX] >= 0.5:
        mask[int(Action.IDENTIFY_USER)] = False

    for action_idx, vec_offset in _CHECK_ACTION_VEC_IDX.items():
        if state_vector[CHECK_START_IDX + vec_offset] >= 0.5:
            mask[action_idx] = False

    for action_idx, vec_offset in _EXPLOIT_ACTION_VEC_IDX.items():
        vector = VECTOR_KEYS[vec_offset]
        if state_vector[FOUND_START_IDX + vec_offset] < 0.5:
            mask[action_idx] = False
            continue

        failure_count = int(
            round(state_vector[EXPLOIT_FAIL_START_IDX + vec_offset] * MAX_EXPLOIT_FAILURES)
        )
        if failure_count >= exploit_retry_budget(vector):
            mask[action_idx] = False

    if state_vector[CURRENT_PRIVILEGE_IDX] < 0.5:
        mask[int(Action.VERIFY_ROOT)] = False

    return mask
