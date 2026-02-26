from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from actions import VECTOR_KEYS


STATE_DIM = 17
CURRENT_PRIVILEGE_IDX = 0
OS_IDENTIFIED_IDX = 1
USER_IDENTIFIED_IDX = 2
CHECK_START_IDX = 3
FOUND_START_IDX = CHECK_START_IDX + len(VECTOR_KEYS)


def _empty_flags() -> Dict[str, bool]:
    return {vector: False for vector in VECTOR_KEYS}


@dataclass
class ObservableState:
    current_privilege: int = 0  # 0=user, 1=root
    os_identified: bool = False
    user_identified: bool = False
    checked: Dict[str, bool] = field(default_factory=_empty_flags)
    found: Dict[str, bool] = field(default_factory=_empty_flags)

    def to_vector(self) -> np.ndarray:
        output = np.zeros(STATE_DIM, dtype=np.float32)
        output[CURRENT_PRIVILEGE_IDX] = float(self.current_privilege)
        output[OS_IDENTIFIED_IDX] = float(self.os_identified)
        output[USER_IDENTIFIED_IDX] = float(self.user_identified)

        for idx, vector in enumerate(VECTOR_KEYS):
            output[CHECK_START_IDX + idx] = float(self.checked[vector])
            output[FOUND_START_IDX + idx] = float(self.found[vector])
        return output

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "ObservableState":
        if vector.shape[0] != STATE_DIM:
            raise ValueError(f"Expected state dimension {STATE_DIM}, got {vector.shape[0]}")

        checked = _empty_flags()
        found = _empty_flags()
        for idx, key in enumerate(VECTOR_KEYS):
            checked[key] = bool(vector[CHECK_START_IDX + idx] >= 0.5)
            found[key] = bool(vector[FOUND_START_IDX + idx] >= 0.5)

        return cls(
            current_privilege=int(vector[CURRENT_PRIVILEGE_IDX] >= 0.5),
            os_identified=bool(vector[OS_IDENTIFIED_IDX] >= 0.5),
            user_identified=bool(vector[USER_IDENTIFIED_IDX] >= 0.5),
            checked=checked,
            found=found,
        )

    def found_vectors(self) -> List[str]:
        return [vector for vector in VECTOR_KEYS if self.found[vector]]

