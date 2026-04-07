"""Deterministic expert baseline policy for the 16-action practical action space."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from postex_agent.core.actions import Action, CHECK_ACTIONS, VECTOR_BY_CHECK_ACTION
from postex_agent.core.state import HostState


VECTOR_TO_EXPLOIT_ACTION: Dict[str, Action] = {
    "sudo": Action.EXPLOIT_SUDO,
    "suid": Action.EXPLOIT_SUID,
    "capabilities": Action.EXPLOIT_CAP,
    "cron": Action.EXPLOIT_CRON,
    "kernel": Action.EXPLOIT_KERNEL,
}

LOWEST_RISK_VECTOR_ORDER: List[str] = [
    "sudo",
    "cron",
    "suid",
    "capabilities",
    "kernel",
]


class BaselinePolicy:
    """
    Competent deterministic baseline:
    1) Identify OS
    2) Identify user
    3) Enumerate all vectors in a fixed order
    4) Exploit the lowest-risk found vector
    5) Verify root if currently root, else stop
    """

    def select_action(self, state_vector: np.ndarray) -> int:
        state = HostState.from_vector(state_vector)

        if state.current_privilege == 1:
            return int(Action.VERIFY_ROOT)

        if not state.os_identified:
            return int(Action.IDENTIFY_OS)

        if not state.user_identified:
            return int(Action.IDENTIFY_USER)

        for action in CHECK_ACTIONS:
            vector = VECTOR_BY_CHECK_ACTION[action]
            if not state.checked[vector]:
                return int(action)

        for vector in LOWEST_RISK_VECTOR_ORDER:
            if state.found.get(vector, False):
                # Skip vectors we already tried and failed
                if state.exploit_failures.get(vector, 0) > 0:
                    continue
                exploit_action = VECTOR_TO_EXPLOIT_ACTION.get(vector)
                if exploit_action is not None:
                    return int(exploit_action)

        return int(Action.STOP)

