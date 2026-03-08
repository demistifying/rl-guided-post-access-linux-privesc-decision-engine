"""
Structured host state representation.
Shared between simulation training and real-session deployment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

VECTOR_KEYS: List[str] = [
    "sudo",
    "suid",
    "capabilities",
    "writable_path",
    "cron",
    "credentials",
    "kernel",
]

STATE_DIM = 17  # 1 privilege + 1 os + 1 user + 7 checked + 7 found

CURRENT_PRIVILEGE_IDX = 0
OS_IDENTIFIED_IDX     = 1
USER_IDENTIFIED_IDX   = 2
CHECK_START_IDX       = 3
FOUND_START_IDX       = CHECK_START_IDX + len(VECTOR_KEYS)


def _empty_flags() -> Dict[str, bool]:
    return {v: False for v in VECTOR_KEYS}


@dataclass
class HostState:
    """Full observable state of the target host."""

    current_privilege: int = 0          # 0 = user, 1 = root
    os_identified: bool = False
    user_identified: bool = False
    checked: Dict[str, bool] = field(default_factory=_empty_flags)
    found: Dict[str, bool] = field(default_factory=_empty_flags)

    # Rich metadata (not in RL vector, used for CLI/logging)
    kernel_version: Optional[str] = None
    current_user: Optional[str] = None
    os_info: Optional[str] = None
    sudo_commands: List[str] = field(default_factory=list)
    exploitable_suid_bins: List[str] = field(default_factory=list)
    credentials_found: List[str] = field(default_factory=list)
    exploitable_caps: List[str] = field(default_factory=list)
    writable_paths: List[str] = field(default_factory=list)
    cron_jobs: List[str] = field(default_factory=list)

    # ---------- vector conversion ----------

    def to_vector(self) -> np.ndarray:
        vec = np.zeros(STATE_DIM, dtype=np.float32)
        vec[CURRENT_PRIVILEGE_IDX] = float(self.current_privilege)
        vec[OS_IDENTIFIED_IDX]     = float(self.os_identified)
        vec[USER_IDENTIFIED_IDX]   = float(self.user_identified)
        for i, key in enumerate(VECTOR_KEYS):
            vec[CHECK_START_IDX + i] = float(self.checked[key])
            vec[FOUND_START_IDX + i] = float(self.found[key])
        return vec

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "HostState":
        if vec.shape[0] != STATE_DIM:
            raise ValueError(f"Expected {STATE_DIM}-dim vector, got {vec.shape[0]}")
        checked = _empty_flags()
        found   = _empty_flags()
        for i, key in enumerate(VECTOR_KEYS):
            checked[key] = bool(vec[CHECK_START_IDX + i] >= 0.5)
            found[key]   = bool(vec[FOUND_START_IDX + i] >= 0.5)
        return cls(
            current_privilege = int(vec[CURRENT_PRIVILEGE_IDX] >= 0.5),
            os_identified     = bool(vec[OS_IDENTIFIED_IDX] >= 0.5),
            user_identified   = bool(vec[USER_IDENTIFIED_IDX] >= 0.5),
            checked=checked,
            found=found,
        )

    # ---------- helpers ----------

    def found_vectors(self) -> List[str]:
        return [v for v in VECTOR_KEYS if self.found[v]]

    def all_checked(self) -> bool:
        return all(self.checked.values())

    def any_found(self) -> bool:
        return any(self.found.values())

    def summary(self) -> str:
        priv = "root" if self.current_privilege == 1 else "user"
        found = self.found_vectors() or ["none"]
        return (
            f"privilege={priv} "
            f"os={self.os_identified} user={self.user_identified} "
            f"found={found}"
        )
