"""
Structured host state representation shared by training and live deployment.

State vector layout (35 dimensions):
  [0]      current_privilege   (0=user, 1=root)
  [1]      os_identified       (0/1)
  [2]      user_identified     (0/1)
  [3-9]    checked[7]          per-vector binary
  [10-16]  found[7]            per-vector binary
  [17-23]  exploit_failures[7] per-vector, normalised 0-1
  [24-30]  richness[7]         per-vector item count, normalised 0-1
  [31]     cred_count          normalised credential count
  [32]     cred_quality        ordinal: 0=none, 0.33=hash, 0.66=plaintext, 1.0=key/root
  [33]     time_step           step / max_steps
  [34]     cumulative_risk     total risk consumed, normalised 0-1
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

STATE_DIM = 35

CURRENT_PRIVILEGE_IDX = 0
OS_IDENTIFIED_IDX = 1
USER_IDENTIFIED_IDX = 2
CHECK_START_IDX = 3
FOUND_START_IDX = CHECK_START_IDX + len(VECTOR_KEYS)
EXPLOIT_FAIL_START_IDX = FOUND_START_IDX + len(VECTOR_KEYS)
RICHNESS_START_IDX = EXPLOIT_FAIL_START_IDX + len(VECTOR_KEYS)
CRED_COUNT_IDX = RICHNESS_START_IDX + len(VECTOR_KEYS)
CRED_QUALITY_IDX = CRED_COUNT_IDX + 1
TIME_STEP_IDX = CRED_QUALITY_IDX + 1
CUMULATIVE_RISK_IDX = TIME_STEP_IDX + 1

MAX_EXPLOIT_FAILURES = 5
MAX_RICHNESS = 10


def _empty_flags() -> Dict[str, bool]:
    return {v: False for v in VECTOR_KEYS}


def _zero_counts() -> Dict[str, int]:
    return {v: 0 for v in VECTOR_KEYS}


def _zero_floats() -> Dict[str, float]:
    return {v: 0.0 for v in VECTOR_KEYS}


@dataclass
class HostState:
    """Full observable state of the target host."""

    current_privilege: int = 0
    os_identified: bool = False
    user_identified: bool = False
    checked: Dict[str, bool] = field(default_factory=_empty_flags)
    found: Dict[str, bool] = field(default_factory=_empty_flags)

    exploit_failures: Dict[str, int] = field(default_factory=_zero_counts)
    richness: Dict[str, float] = field(default_factory=_zero_floats)
    cred_count: float = 0.0
    cred_quality: float = 0.0
    time_step: float = 0.0
    cumulative_risk: float = 0.0

    # Observable metadata used by the live assistant and logs.
    kernel_version: Optional[str] = None
    current_user: Optional[str] = None
    os_info: Optional[str] = None
    sudo_commands: List[str] = field(default_factory=list)
    sudo_nopasswd_entries: List[str] = field(default_factory=list)
    sudo_password_entries: List[str] = field(default_factory=list)
    exploitable_suid_bins: List[str] = field(default_factory=list)
    credentials_found: List[str] = field(default_factory=list)
    exploitable_caps: List[str] = field(default_factory=list)
    writable_paths: List[str] = field(default_factory=list)
    cron_jobs: List[str] = field(default_factory=list)
    cron_writable_targets: List[str] = field(default_factory=list)
    user_groups: List[str] = field(default_factory=list)
    privileged_groups: List[str] = field(default_factory=list)
    container_indicators: List[str] = field(default_factory=list)
    is_containerized: bool = False

    def to_vector(self) -> np.ndarray:
        vec = np.zeros(STATE_DIM, dtype=np.float32)

        vec[CURRENT_PRIVILEGE_IDX] = float(self.current_privilege)
        vec[OS_IDENTIFIED_IDX] = float(self.os_identified)
        vec[USER_IDENTIFIED_IDX] = float(self.user_identified)

        for i, key in enumerate(VECTOR_KEYS):
            vec[CHECK_START_IDX + i] = float(self.checked[key])
            vec[FOUND_START_IDX + i] = float(self.found[key])

        for i, key in enumerate(VECTOR_KEYS):
            vec[EXPLOIT_FAIL_START_IDX + i] = min(
                self.exploit_failures[key] / MAX_EXPLOIT_FAILURES,
                1.0,
            )

        for i, key in enumerate(VECTOR_KEYS):
            vec[RICHNESS_START_IDX + i] = min(
                self.richness[key] / MAX_RICHNESS,
                1.0,
            )

        vec[CRED_COUNT_IDX] = min(self.cred_count, 1.0)
        vec[CRED_QUALITY_IDX] = min(self.cred_quality, 1.0)
        vec[TIME_STEP_IDX] = self.time_step
        vec[CUMULATIVE_RISK_IDX] = min(self.cumulative_risk, 1.0)
        return vec

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "HostState":
        if vec.shape[0] != STATE_DIM:
            raise ValueError(f"Expected {STATE_DIM}-dim vector, got {vec.shape[0]}")

        checked = _empty_flags()
        found = _empty_flags()
        exploit_failures = _zero_counts()
        richness = _zero_floats()

        for i, key in enumerate(VECTOR_KEYS):
            checked[key] = bool(vec[CHECK_START_IDX + i] >= 0.5)
            found[key] = bool(vec[FOUND_START_IDX + i] >= 0.5)
            exploit_failures[key] = int(
                round(vec[EXPLOIT_FAIL_START_IDX + i] * MAX_EXPLOIT_FAILURES)
            )
            richness[key] = float(vec[RICHNESS_START_IDX + i] * MAX_RICHNESS)

        return cls(
            current_privilege=int(vec[CURRENT_PRIVILEGE_IDX] >= 0.5),
            os_identified=bool(vec[OS_IDENTIFIED_IDX] >= 0.5),
            user_identified=bool(vec[USER_IDENTIFIED_IDX] >= 0.5),
            checked=checked,
            found=found,
            exploit_failures=exploit_failures,
            richness=richness,
            cred_count=float(vec[CRED_COUNT_IDX]),
            cred_quality=float(vec[CRED_QUALITY_IDX]),
            time_step=float(vec[TIME_STEP_IDX]),
            cumulative_risk=float(vec[CUMULATIVE_RISK_IDX]),
        )

    def found_vectors(self) -> List[str]:
        return [v for v in VECTOR_KEYS if self.found[v]]

    def all_checked(self) -> bool:
        return all(self.checked.values())

    def any_found(self) -> bool:
        return any(self.found.values())

    def summary(self) -> str:
        privilege = "root" if self.current_privilege == 1 else "user"
        found = self.found_vectors() or ["none"]
        container = " containerized=True" if self.is_containerized else ""
        return (
            f"privilege={privilege} "
            f"os={self.os_identified} user={self.user_identified} "
            f"found={found}{container}"
        )
