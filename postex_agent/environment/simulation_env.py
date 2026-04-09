"""Simulation environment for RL training.

Track 3 replaces independent vector sampling with hidden host archetypes,
vector-specific exploit success rates, and a limited retry budget for genuinely
stochastic kernel exploitation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from postex_agent.core.actions import (
    ACTION_SPACE_SIZE,
    Action,
    VECTOR_BY_CHECK_ACTION,
    VECTOR_BY_EXPLOIT_ACTION,
    compute_action_mask,
    exploit_retry_budget,
)
from postex_agent.core.state import HostState, MAX_RICHNESS, VECTOR_KEYS
from postex_agent.environment.host_archetypes import (
    ARCHETYPE_TEMPLATES,
    DEFAULT_DETERMINISTIC_SUCCESS,
    SampledArchetypeProfile,
    sample_archetype_profile,
)


VECTOR_DISCOVERY_PROBS: Dict[str, float] = {
    key: sum(template.weight * template.vector_probs.get(key, 0.0) for template in ARCHETYPE_TEMPLATES)
    for key in VECTOR_KEYS
}

VECTOR_SUCCESS_PROBS: Dict[str, float] = dict(DEFAULT_DETERMINISTIC_SUCCESS)

VECTOR_RISK_PENALTIES: Dict[str, float] = {
    "sudo": 0.10,
    "suid": 0.20,
    "capabilities": 0.30,
    "writable_path": 0.25,
    "cron": 0.18,
    "credentials": 0.22,
    "kernel": 0.45,
}

R_STEP = -0.15
R_USEFUL_ENUM = 0.5
R_REDUNDANT = -1.0
R_DISCOVERY = 1.5
R_SUCCESS_ESCALATION = 18.0
R_FAILED_ESCALATION = -4.0
R_RETRYABLE_FAILURE = -2.0
R_PREMATURE_ESCALATION = -2.5
R_EXPLOIT_WITHOUT_ENUM = -2.5
R_CORRECT_STOP = 0.25
R_PREMATURE_STOP = -6.0
R_UNSAFE_KERNEL_ATTEMPT = -1.5

MAX_EPISODE_STEPS = 20
MAX_CUMULATIVE_RISK = sum(VECTOR_RISK_PENALTIES.values())

ACTIONABLE_VECTORS: List[str] = list(dict.fromkeys(VECTOR_BY_EXPLOIT_ACTION.values()))
SAFER_THAN_KERNEL: List[str] = ["sudo", "suid", "capabilities", "cron"]
HIGH_CONFIDENCE_VECTORS: List[str] = ["sudo", "suid", "capabilities"]

StepResult = Tuple[np.ndarray, float, bool, Dict[str, Any]]


@dataclass
class SimulatedHost:
    """Hidden ground truth for one simulated target."""

    archetype: str
    profile: SampledArchetypeProfile
    vectors: Dict[str, bool] = field(default_factory=dict)
    item_counts: Dict[str, int] = field(default_factory=dict)
    success_probs: Dict[str, float] = field(default_factory=dict)
    cron_chain_success: float = 0.85
    cred_type_quality: float = 0.0
    cred_type_count: int = 0

    def is_viable(self, vector: str) -> bool:
        return self.vectors.get(vector, False)

    def any_viable(self) -> bool:
        return any(self.vectors.values())

    def any_actionable_exploit(self) -> bool:
        return any(self.vectors.get(vector, False) for vector in ACTIONABLE_VECTORS)

    def remaining_hidden_paths(self, state: HostState) -> List[str]:
        remaining: List[str] = []
        for vector in ACTIONABLE_VECTORS:
            if not self.vectors.get(vector, False):
                continue
            if state.exploit_failures.get(vector, 0) >= exploit_retry_budget(vector):
                continue
            remaining.append(vector)
        return remaining


def _sample_item_count(rng: random.Random, mean: float) -> int:
    spread = max(mean * 0.35, 0.5)
    return max(1, min(int(round(rng.gauss(mean, spread))), MAX_RICHNESS))


def _sample_credential_quality(
    rng: random.Random,
    levels: Tuple[Tuple[float, float], ...],
) -> float:
    roll = rng.random()
    cumulative = 0.0
    for quality, probability in levels:
        cumulative += probability
        if roll <= cumulative:
            return quality
    return levels[-1][0] if levels else 0.0


def _sample_host(
    rng: random.Random,
    archetype_name: Optional[str] = None,
) -> SimulatedHost:
    profile = sample_archetype_profile(rng, force_name=archetype_name)
    vectors = {
        vector: rng.random() < profile.vector_probs.get(vector, 0.0)
        for vector in VECTOR_KEYS
    }

    item_counts: Dict[str, int] = {}
    for vector in VECTOR_KEYS:
        if vectors[vector]:
            item_counts[vector] = _sample_item_count(rng, profile.richness_means.get(vector, 1.0))
        else:
            item_counts[vector] = 0

    cred_quality = 0.0
    cred_count = 0
    if vectors.get("credentials", False):
        cred_count = item_counts["credentials"]
        cred_quality = _sample_credential_quality(rng, profile.cred_quality_levels)

    return SimulatedHost(
        archetype=profile.name,
        profile=profile,
        vectors=vectors,
        item_counts=item_counts,
        success_probs=dict(profile.success_probs),
        cron_chain_success=profile.cron_chain_success,
        cred_type_quality=cred_quality,
        cred_type_count=cred_count,
    )


def _cron_chain_active(state: HostState) -> bool:
    return state.found.get("cron", False) and state.found.get("writable_path", False)


def _safer_path_available(state: HostState) -> bool:
    for vector in SAFER_THAN_KERNEL:
        if not state.found.get(vector, False):
            continue
        if vector == "cron" and not _cron_chain_active(state):
            continue
        if state.exploit_failures.get(vector, 0) >= exploit_retry_budget(vector):
            continue
        return True
    return False


def _high_confidence_path_available(state: HostState) -> bool:
    for vector in HIGH_CONFIDENCE_VECTORS:
        if not state.found.get(vector, False):
            continue
        if state.exploit_failures.get(vector, 0) >= exploit_retry_budget(vector):
            continue
        return True
    return False


def _attempt_escalation(
    host: SimulatedHost,
    state: HostState,
    vector: str,
    rng: random.Random,
) -> Tuple[bool, Dict[str, Any]]:
    if not host.is_viable(vector):
        return False, {"success_prob": 0.0, "chain_used": False}

    chain_used = False
    success_prob = host.success_probs.get(vector, VECTOR_SUCCESS_PROBS.get(vector, 0.0))

    if vector == "capabilities":
        if state.richness.get(vector, 0.0) >= 0.5:
            success_prob = min(success_prob + 0.05, 0.97)
    elif vector == "cron":
        if _cron_chain_active(state):
            success_prob = host.cron_chain_success
            chain_used = True
    elif vector == "kernel":
        if state.os_identified:
            success_prob = min(success_prob + 0.05, 0.80)
            chain_used = True

    return rng.random() < success_prob, {
        "success_prob": success_prob,
        "chain_used": chain_used,
    }


class SimulationEnv:
    """Gym-style environment for DQN training."""

    def __init__(self, seed: Optional[int] = None, max_steps: int = MAX_EPISODE_STEPS):
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self.state: HostState = HostState()
        self.host: Optional[SimulatedHost] = None
        self._steps = 0
        self._done = False
        self._ep_reward = 0.0
        self._risk_exp = 0.0
        self._redundant = 0
        self._esc_attempts = 0
        self._done_reason = ""
        self._stop_was_correct = False
        self._kernel_attempted_while_safer_path_available = False

    def reset(
        self,
        seed: Optional[int] = None,
        archetype_name: Optional[str] = None,
    ) -> np.ndarray:
        if seed is not None:
            self._rng = random.Random(seed)
        self.state = HostState()
        self.host = _sample_host(self._rng, archetype_name=archetype_name)
        self._steps = 0
        self._done = False
        self._ep_reward = 0.0
        self._risk_exp = 0.0
        self._redundant = 0
        self._esc_attempts = 0
        self._done_reason = ""
        self._stop_was_correct = False
        self._kernel_attempted_while_safer_path_available = False
        state_vec = self.state.to_vector()
        self._current_mask = compute_action_mask(state_vec)
        return state_vec

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("Call reset() before step().")
        if self.host is None:
            raise RuntimeError("Call reset() first.")

        action_enum = Action(action)
        reward = R_STEP
        high_confidence_path_available = _high_confidence_path_available(self.state)
        info: Dict[str, Any] = {
            "action": action_enum.name,
            "archetype": self.host.archetype,
        }

        if action_enum == Action.IDENTIFY_OS:
            if self.state.os_identified:
                reward += R_REDUNDANT
                self._redundant += 1
            else:
                self.state.os_identified = True
                reward += R_USEFUL_ENUM

        elif action_enum == Action.IDENTIFY_USER:
            if self.state.user_identified:
                reward += R_REDUNDANT
                self._redundant += 1
            else:
                self.state.user_identified = True
                reward += R_USEFUL_ENUM

        elif action_enum in VECTOR_BY_CHECK_ACTION:
            vector = VECTOR_BY_CHECK_ACTION[action_enum]
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

                    raw_count = self.host.item_counts.get(vector, 0)
                    self.state.richness[vector] = min(raw_count / MAX_RICHNESS, 1.0)

                    if vector == "credentials":
                        self.state.cred_count = min(
                            self.host.cred_type_count / MAX_RICHNESS, 1.0
                        )
                        self.state.cred_quality = self.host.cred_type_quality

        elif action_enum in VECTOR_BY_EXPLOIT_ACTION:
            vector = VECTOR_BY_EXPLOIT_ACTION[action_enum]
            if not self.state.checked[vector]:
                reward += R_EXPLOIT_WITHOUT_ENUM
                info["unenumerated_exploit"] = True

            if not self.state.found[vector]:
                reward += R_PREMATURE_ESCALATION
                info["premature_escalation"] = True
            else:
                risk = VECTOR_RISK_PENALTIES[vector]
                reward -= risk
                self._risk_exp += risk
                self._esc_attempts += 1

                if vector == "kernel" and _safer_path_available(self.state):
                    reward += R_UNSAFE_KERNEL_ATTEMPT
                    self._kernel_attempted_while_safer_path_available = True
                    info["safer_path_available"] = True
                    info["unsafe_kernel_penalty"] = True

                success, details = _attempt_escalation(
                    self.host,
                    self.state,
                    vector,
                    self._rng,
                )
                info["escalation_vector"] = vector
                info["escalation_success"] = success
                info.update(details)

                if success:
                    self.state.current_privilege = 1
                    reward += R_SUCCESS_ESCALATION
                else:
                    self.state.exploit_failures[vector] += 1
                    if exploit_retry_budget(vector) > 1:
                        reward += R_RETRYABLE_FAILURE
                    else:
                        reward += R_FAILED_ESCALATION

        elif action_enum == Action.VERIFY_ROOT:
            if self.state.current_privilege == 1:
                reward += R_USEFUL_ENUM
            else:
                reward += R_REDUNDANT

        elif action_enum == Action.STOP:
            remaining_paths = self.host.remaining_hidden_paths(self.state)
            info["remaining_hidden_paths"] = remaining_paths
            if remaining_paths:
                reward += R_PREMATURE_STOP
                self._stop_was_correct = False
            else:
                reward += R_CORRECT_STOP
                self._stop_was_correct = True
            self._done = True
            self._done_reason = "stop"

        if (
            high_confidence_path_available
            and action_enum not in VECTOR_BY_EXPLOIT_ACTION
            and action_enum not in (Action.VERIFY_ROOT, Action.STOP)
        ):
            reward += R_UNSAFE_KERNEL_ATTEMPT / 2.0
            info["delayed_high_confidence_exploit"] = True

        self._steps += 1
        self._ep_reward += reward

        self.state.time_step = self._steps / self.max_steps
        self.state.cumulative_risk = min(self._risk_exp / MAX_CUMULATIVE_RISK, 1.0)

        if self.state.current_privilege == 1:
            self._done = True
            self._done_reason = "root"
        elif self._steps >= self.max_steps:
            self._done = True
            self._done_reason = "max_steps"

        if self._done:
            info["episode"] = self._episode_metrics()

        state_vec = self.state.to_vector()
        self._current_mask = compute_action_mask(state_vec)
        info["action_mask"] = self._current_mask
        return state_vec, reward, self._done, info

    def _episode_metrics(self) -> Dict[str, Any]:
        archetype = self.host.archetype if self.host else "unknown"
        had_actionable_path = self.host.any_actionable_exploit() if self.host else False
        return {
            "success": self.state.current_privilege == 1,
            "steps": self._steps,
            "reward": self._ep_reward,
            "risk_exposure": self._risk_exp,
            "redundant_actions": self._redundant,
            "escalation_attempts": self._esc_attempts,
            "archetype": archetype,
            "done_reason": self._done_reason,
            "stop_episode": self._done_reason == "stop",
            "stop_correct": self._stop_was_correct,
            "kernel_attempted_while_safer_path_available": (
                self._kernel_attempted_while_safer_path_available
            ),
            "had_actionable_path": had_actionable_path,
        }

    @property
    def observation_space_size(self) -> int:
        from postex_agent.core.state import STATE_DIM

        return STATE_DIM

    @property
    def action_space_size(self) -> int:
        return ACTION_SPACE_SIZE

    def action_mask(self) -> np.ndarray:
        if hasattr(self, "_current_mask"):
            return self._current_mask
        return compute_action_mask(self.state.to_vector())
