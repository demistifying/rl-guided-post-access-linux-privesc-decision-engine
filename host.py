from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict

from actions import VECTOR_KEYS


VECTOR_DISCOVERY_PROBABILITIES: Dict[str, float] = {
    "sudo": 0.25,
    "suid": 0.35,
    "capabilities": 0.15,
    "writable_path": 0.30,
    "cron": 0.20,
    "credentials": 0.20,
    "kernel": 0.10,
}


VECTOR_SUCCESS_PROBABILITIES: Dict[str, float] = {
    "sudo": 0.95,
    "suid": 0.80,
    "capabilities": 0.75,
    "writable_path": 0.70,
    "cron": 0.85,
    "credentials": 0.75,
    "kernel": 0.60,
}


VECTOR_RISK_PENALTIES: Dict[str, float] = {
    "sudo": 0.10,
    "suid": 0.20,
    "capabilities": 0.30,
    "writable_path": 0.25,
    "cron": 0.18,
    "credentials": 0.22,
    "kernel": 0.45,
}


LOWEST_RISK_VECTOR_ORDER = sorted(
    VECTOR_KEYS, key=lambda vector: VECTOR_RISK_PENALTIES[vector]
)


@dataclass(frozen=True)
class HostGroundTruth:
    vectors: Dict[str, bool]

    def is_viable(self, vector: str) -> bool:
        return self.vectors.get(vector, False)


def sample_host(rng: random.Random) -> HostGroundTruth:
    vectors: Dict[str, bool] = {}
    for vector in VECTOR_KEYS:
        vectors[vector] = rng.random() < VECTOR_DISCOVERY_PROBABILITIES[vector]
    return HostGroundTruth(vectors=vectors)


def attempt_escalation(host: HostGroundTruth, vector: str, rng: random.Random) -> bool:
    if not host.is_viable(vector):
        return False
    return rng.random() < VECTOR_SUCCESS_PROBABILITIES[vector]

