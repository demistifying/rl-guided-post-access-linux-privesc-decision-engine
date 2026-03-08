"""
Simulation environment for RL training.
Models Linux privilege escalation as a finite episodic MDP.
No real commands are executed here – host properties are sampled stochastically.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from postex_agent.core.actions import (
    Action,
    ACTION_SPACE_SIZE,
    VECTOR_BY_CHECK_ACTION,
    VECTOR_BY_EXPLOIT_ACTION,
)
from postex_agent.core.state import HostState, VECTOR_KEYS


# ─── Host ground-truth probabilities ────────────────────────────────────────

VECTOR_DISCOVERY_PROBS: Dict[str, float] = {
    "sudo":         0.25,
    "suid":         0.35,
    "capabilities": 0.15,
    "writable_path": 0.30,
    "cron":         0.20,
    "credentials":  0.20,
    "kernel":       0.10,
}

VECTOR_SUCCESS_PROBS: Dict[str, float] = {
    "sudo":         0.95,
    "suid":         0.80,
    "capabilities": 0.75,
    "writable_path": 0.70,
    "cron":         0.85,
    "credentials":  0.75,
    "kernel":       0.60,
}

VECTOR_RISK_PENALTIES: Dict[str, float] = {
    "sudo":         0.10,
    "suid":         0.20,
    "capabilities": 0.30,
    "writable_path": 0.25,
    "cron":         0.18,
    "credentials":  0.22,
    "kernel":       0.45,
}

LOWEST_RISK_ORDER: List[str] = sorted(
    VECTOR_KEYS, key=lambda v: VECTOR_RISK_PENALTIES[v]
)

# ─── Reward constants ────────────────────────────────────────────────────────

R_STEP                   = -0.1
R_USEFUL_ENUM            =  1.0
R_REDUNDANT              = -1.0
R_DISCOVERY              =  3.0
R_SUCCESS_ESCALATION     = 10.0
R_FAILED_ESCALATION      = -3.0
R_PREMATURE_ESCALATION   = -2.0
R_EXPLOIT_WITHOUT_ENUM   = -2.0

MAX_EPISODE_STEPS = 20

StepResult = Tuple[np.ndarray, float, bool, Dict[str, Any]]


@dataclass
class SimulatedHost:
    """Hidden ground truth for one simulated target."""
    vectors: Dict[str, bool] = field(default_factory=dict)

    def is_viable(self, vector: str) -> bool:
        return self.vectors.get(vector, False)

    def any_viable(self) -> bool:
        return any(self.vectors.values())


def _sample_host(rng: random.Random) -> SimulatedHost:
    vectors = {
        v: rng.random() < VECTOR_DISCOVERY_PROBS[v]
        for v in VECTOR_KEYS
    }
    # Guarantee at least one viable vector (avoids unsolvable episodes)
    if not any(vectors.values()):
        vectors[rng.choice(VECTOR_KEYS)] = True
    return SimulatedHost(vectors=vectors)


def _attempt_escalation(host: SimulatedHost, vector: str, rng: random.Random) -> bool:
    if not host.is_viable(vector):
        return False
    return rng.random() < VECTOR_SUCCESS_PROBS[vector]


def _pick_best_vector(state: HostState) -> Optional[str]:
    """Choose lowest-risk found vector, prefer those already checked."""
    found = set(state.found_vectors())
    for v in LOWEST_RISK_ORDER:
        if v in found:
            return v
    return None


class SimulationEnv:
    """
    Gym-style environment for DQN training.

    observation_space: np.ndarray of shape (STATE_DIM,)
    action_space:      discrete, size = ACTION_SPACE_SIZE
    """

    def __init__(self, seed: Optional[int] = None, max_steps: int = MAX_EPISODE_STEPS):
        self.max_steps = max_steps
        self._rng      = random.Random(seed)
        self.state:    HostState          = HostState()
        self.host:     Optional[SimulatedHost] = None
        self._steps    = 0
        self._done     = False
        self._ep_reward   = 0.0
        self._risk_exp    = 0.0
        self._redundant   = 0
        self._esc_attempts = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = random.Random(seed)
        self.state          = HostState()
        self.host           = _sample_host(self._rng)
        self._steps         = 0
        self._done          = False
        self._ep_reward     = 0.0
        self._risk_exp      = 0.0
        self._redundant     = 0
        self._esc_attempts  = 0
        return self.state.to_vector()

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("Call reset() before step().")
        if self.host is None:
            raise RuntimeError("Call reset() first.")

        a      = Action(action)
        reward = R_STEP
        info: Dict[str, Any] = {"action": a.name}

        # ── Identification ────────────────────────────────────────────────
        if a == Action.IDENTIFY_OS:
            if self.state.os_identified:
                reward += R_REDUNDANT
                self._redundant += 1
            else:
                self.state.os_identified = True
                reward += R_USEFUL_ENUM

        elif a == Action.IDENTIFY_USER:
            if self.state.user_identified:
                reward += R_REDUNDANT
                self._redundant += 1
            else:
                self.state.user_identified = True
                reward += R_USEFUL_ENUM

        # ── Enumeration ───────────────────────────────────────────────────
        elif a in VECTOR_BY_CHECK_ACTION:
            vector = VECTOR_BY_CHECK_ACTION[a]
            if self.state.checked[vector]:
                reward += R_REDUNDANT
                self._redundant += 1
                info["redundant"] = True
            else:
                self.state.checked[vector] = True
                reward += R_USEFUL_ENUM
                if self.host.is_viable(vector):
                    self.state.found[vector] = True
                    reward += R_DISCOVERY
                    info["discovered_vector"] = vector

        # ── Exploitation ──────────────────────────────────────────────────
        elif a in VECTOR_BY_EXPLOIT_ACTION:
            vector = VECTOR_BY_EXPLOIT_ACTION[a]
            # Penalize if not yet enumerated
            if not self.state.checked[vector]:
                reward += R_EXPLOIT_WITHOUT_ENUM
                info["unenumerated_exploit"] = True

            if not self.state.found[vector]:
                reward += R_PREMATURE_ESCALATION
                info["premature_escalation"] = True
            else:
                risk  = VECTOR_RISK_PENALTIES[vector]
                reward -= risk
                self._risk_exp    += risk
                self._esc_attempts += 1

                success = _attempt_escalation(self.host, vector, self._rng)
                info["escalation_vector"]  = vector
                info["escalation_success"] = success

                if success:
                    self.state.current_privilege = 1
                    reward += R_SUCCESS_ESCALATION
                else:
                    reward += R_FAILED_ESCALATION

        # ── Generic ESCALATE (finds best vector automatically) ────────────
        elif a == Action.ESCALATE if hasattr(Action, "ESCALATE") else False:
            vector = _pick_best_vector(self.state)
            if vector is None:
                reward += R_PREMATURE_ESCALATION
            else:
                risk   = VECTOR_RISK_PENALTIES[vector]
                reward -= risk
                self._risk_exp    += risk
                self._esc_attempts += 1
                success = _attempt_escalation(self.host, vector, self._rng)
                info["escalation_vector"]  = vector
                info["escalation_success"] = success
                if success:
                    self.state.current_privilege = 1
                    reward += R_SUCCESS_ESCALATION
                else:
                    reward += R_FAILED_ESCALATION

        # ── Verify Root ───────────────────────────────────────────────────
        elif a == Action.VERIFY_ROOT:
            if self.state.current_privilege == 1:
                reward += R_USEFUL_ENUM
            else:
                reward += R_REDUNDANT

        # ── Stop ──────────────────────────────────────────────────────────
        elif a == Action.STOP:
            self._done = True

        self._steps     += 1
        self._ep_reward += reward

        if self.state.current_privilege == 1:
            self._done = True
        if self._steps >= self.max_steps:
            self._done = True

        if self._done:
            info["episode"] = self._episode_metrics()

        return self.state.to_vector(), reward, self._done, info

    # ── Metrics ────────────────────────────────────────────────────────────

    def _episode_metrics(self) -> Dict[str, Any]:
        return {
            "success":             self.state.current_privilege == 1,
            "steps":               self._steps,
            "reward":              self._ep_reward,
            "risk_exposure":       self._risk_exp,
            "redundant_actions":   self._redundant,
            "escalation_attempts": self._esc_attempts,
        }

    @property
    def observation_space_size(self) -> int:
        from postex_agent.core.state import STATE_DIM
        return STATE_DIM

    @property
    def action_space_size(self) -> int:
        return ACTION_SPACE_SIZE
