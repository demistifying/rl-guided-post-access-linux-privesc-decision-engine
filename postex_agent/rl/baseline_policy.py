"""Deterministic baseline policy for the Track 3 simulator."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from postex_agent.core.actions import (
    Action,
    VECTOR_BY_CHECK_ACTION,
    exploit_retry_budget,
)
from postex_agent.core.state import HostState


VECTOR_TO_EXPLOIT_ACTION: Dict[str, Action] = {
    "sudo": Action.EXPLOIT_SUDO,
    "suid": Action.EXPLOIT_SUID,
    "capabilities": Action.EXPLOIT_CAP,
    "cron": Action.EXPLOIT_CRON,
    "kernel": Action.EXPLOIT_KERNEL,
    "credentials": Action.EXPLOIT_CREDENTIALS,
    "writable_path": Action.EXPLOIT_WRITABLE,
}

LOWEST_RISK_VECTOR_ORDER: List[str] = [
    "sudo",
    "suid",
    "capabilities",
    "cron",
    "credentials",
    "writable_path",
    "kernel",
]

CHECK_ORDER: List[Action] = [
    Action.CHECK_SUDO,
    Action.CHECK_SUID,
    Action.CHECK_CAPABILITIES,
    Action.CHECK_CRON,
    Action.CHECK_WRITABLE,
    Action.CHECK_KERNEL,
    Action.SEARCH_CREDENTIALS,
]


def _can_retry(state: HostState, vector: str) -> bool:
    return state.exploit_failures.get(vector, 0) < exploit_retry_budget(vector)


class BaselinePolicy:
    """A competent baseline for Track 3.

    The baseline behaves like a careful operator:
    1. identify OS and user
    2. exploit high-confidence vectors as soon as they are discovered
    3. check writable paths before attempting cron
    4. reserve kernel retries for when safer paths are exhausted
    5. stop once realistic options are exhausted
    """

    def select_action(self, state_vector: np.ndarray) -> int:
        state = HostState.from_vector(state_vector)

        if state.current_privilege == 1:
            return int(Action.VERIFY_ROOT)

        if not state.os_identified:
            return int(Action.IDENTIFY_OS)

        if not state.user_identified:
            return int(Action.IDENTIFY_USER)

        if state.found.get("sudo", False) and _can_retry(state, "sudo"):
            return int(Action.EXPLOIT_SUDO)

        if state.found.get("suid", False) and _can_retry(state, "suid"):
            return int(Action.EXPLOIT_SUID)

        if state.found.get("capabilities", False) and _can_retry(state, "capabilities"):
            return int(Action.EXPLOIT_CAP)

        if state.found.get("cron", False):
            if not state.checked.get("writable_path", False):
                return int(Action.CHECK_WRITABLE)
            if _can_retry(state, "cron"):
                return int(Action.EXPLOIT_CRON)

        # Credentials: try su root if we have high-quality creds
        if state.found.get("credentials", False) and _can_retry(state, "credentials"):
            return int(Action.EXPLOIT_CREDENTIALS)

        for action in CHECK_ORDER:
            vector = VECTOR_BY_CHECK_ACTION[action]
            if not state.checked[vector]:
                return int(action)

        # Writable PATH exploitation (lower confidence, try before kernel)
        if state.found.get("writable_path", False) and _can_retry(state, "writable_path"):
            return int(Action.EXPLOIT_WRITABLE)

        if state.found.get("kernel", False) and _can_retry(state, "kernel"):
            return int(Action.EXPLOIT_KERNEL)

        return int(Action.STOP)
