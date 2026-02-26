from __future__ import annotations

import numpy as np

from actions import Action, CHECK_ACTIONS, VECTOR_BY_CHECK_ACTION
from state import ObservableState


class DeterministicBaselinePolicy:
    """
    Competent deterministic heuristic:
    1) Identify system
    2) Identify user
    3) Enumerate all vectors in fixed order
    4) Escalate if any vector is found, otherwise stop
    """

    def select_action(self, state_vector: np.ndarray) -> int:
        state = ObservableState.from_vector(state_vector)

        if state.current_privilege == 1:
            return int(Action.STOP)

        if not state.os_identified:
            return int(Action.SYSTEM_IDENTIFICATION)

        if not state.user_identified:
            return int(Action.USER_IDENTIFICATION)

        for action in CHECK_ACTIONS:
            vector = VECTOR_BY_CHECK_ACTION[action]
            if not state.checked[vector]:
                return int(action)

        if any(state.found.values()):
            return int(Action.ESCALATE)
        return int(Action.STOP)

