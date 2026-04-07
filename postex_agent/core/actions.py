"""
Action space definition shared between RL training and real deployment.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Dict, List

import numpy as np

from postex_agent.core.state import (
    STATE_DIM,
    CURRENT_PRIVILEGE_IDX,
    OS_IDENTIFIED_IDX,
    USER_IDENTIFIED_IDX,
    CHECK_START_IDX,
    FOUND_START_IDX,
    EXPLOIT_FAIL_START_IDX,
    VECTOR_KEYS,
)


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


# ── Action masking ────────────────────────────────────────────────────────

# Pre-built lookup: CHECK action index → offset into VECTOR_KEYS
_CHECK_ACTION_VEC_IDX: Dict[int, int] = {
    int(a): VECTOR_KEYS.index(v)
    for a, v in VECTOR_BY_CHECK_ACTION.items()
}

# Pre-built lookup: EXPLOIT action index → offset into VECTOR_KEYS
_EXPLOIT_ACTION_VEC_IDX: Dict[int, int] = {
    int(a): VECTOR_KEYS.index(v)
    for a, v in VECTOR_BY_EXPLOIT_ACTION.items()
}


def compute_action_mask(state_vector: np.ndarray) -> np.ndarray:
    """Return a boolean mask of shape (ACTION_SPACE_SIZE,).

    mask[i] = True  → action i is valid in the current state
    mask[i] = False → action i would be redundant or impossible
    """
    mask = np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)

    # IDENTIFY_OS invalid if already identified
    if state_vector[OS_IDENTIFIED_IDX] >= 0.5:
        mask[int(Action.IDENTIFY_OS)] = False

    # IDENTIFY_USER invalid if already identified
    if state_vector[USER_IDENTIFIED_IDX] >= 0.5:
        mask[int(Action.IDENTIFY_USER)] = False

    # CHECK_* invalid if already checked
    for action_idx, vec_offset in _CHECK_ACTION_VEC_IDX.items():
        if state_vector[CHECK_START_IDX + vec_offset] >= 0.5:
            mask[action_idx] = False

    # EXPLOIT_* invalid if vector not found OR already tried and failed
    for action_idx, vec_offset in _EXPLOIT_ACTION_VEC_IDX.items():
        if state_vector[FOUND_START_IDX + vec_offset] < 0.5:
            mask[action_idx] = False
        elif state_vector[EXPLOIT_FAIL_START_IDX + vec_offset] > 0.0:
            # Already attempted this exploit and it failed — don't retry
            mask[action_idx] = False

    # VERIFY_ROOT invalid if not root
    if state_vector[CURRENT_PRIVILEGE_IDX] < 0.5:
        mask[int(Action.VERIFY_ROOT)] = False

    # STOP always valid (already True)
    return mask

