from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np

from actions import Action, VECTOR_BY_CHECK_ACTION
from host import (
    LOWEST_RISK_VECTOR_ORDER,
    VECTOR_RISK_PENALTIES,
    HostGroundTruth,
    attempt_escalation,
    sample_host,
)
from state import ObservableState


SUCCESS_ESCALATION_REWARD = 10.0
DISCOVERY_REWARD = 3.0
USEFUL_ENUM_REWARD = 1.0
REDUNDANT_CHECK_PENALTY = -1.0
PREMATURE_ESCALATION_PENALTY = -2.0
FAILED_ESCALATION_PENALTY = -3.0
STEP_PENALTY = -0.1
MAX_EPISODE_STEPS = 20


StepResult = Tuple[np.ndarray, float, bool, Dict[str, Any]]


class PrivilegeEscalationEnv:
    def __init__(self, seed: Optional[int] = None, max_steps: int = MAX_EPISODE_STEPS):
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self._last_seed = seed

        self.state = ObservableState()
        self.host: Optional[HostGroundTruth] = None
        self._steps = 0
        self._done = False
        self._episode_reward = 0.0
        self._risk_exposure = 0.0
        self._redundant_actions = 0
        self._escalation_attempts = 0

    def seed(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._last_seed = seed

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.seed(seed)

        self.state = ObservableState()
        self.host = sample_host(self._rng)
        self._steps = 0
        self._done = False
        self._episode_reward = 0.0
        self._risk_exposure = 0.0
        self._redundant_actions = 0
        self._escalation_attempts = 0
        return self.state.to_vector()

    def _pick_lowest_risk_found_vector(self) -> Optional[str]:
        found = set(self.state.found_vectors())
        for vector in LOWEST_RISK_VECTOR_ORDER:
            if vector in found:
                return vector
        return None

    def _episode_metrics(self) -> Dict[str, Any]:
        return {
            "success": self.state.current_privilege == 1,
            "steps": self._steps,
            "reward": self._episode_reward,
            "risk_exposure": self._risk_exposure,
            "redundant_actions": self._redundant_actions,
            "escalation_attempts": self._escalation_attempts,
        }

    def step(self, action: int | Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before calling step() again.")
        if self.host is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        action_enum = Action(int(action))
        reward = STEP_PENALTY
        info: Dict[str, Any] = {"action": action_enum.name}

        if action_enum == Action.SYSTEM_IDENTIFICATION:
            if self.state.os_identified:
                reward += REDUNDANT_CHECK_PENALTY
                self._redundant_actions += 1
                info["redundant"] = True
            else:
                self.state.os_identified = True
                reward += USEFUL_ENUM_REWARD

        elif action_enum == Action.USER_IDENTIFICATION:
            if self.state.user_identified:
                reward += REDUNDANT_CHECK_PENALTY
                self._redundant_actions += 1
                info["redundant"] = True
            else:
                self.state.user_identified = True
                reward += USEFUL_ENUM_REWARD

        elif action_enum in VECTOR_BY_CHECK_ACTION:
            vector = VECTOR_BY_CHECK_ACTION[action_enum]
            if self.state.checked[vector]:
                reward += REDUNDANT_CHECK_PENALTY
                self._redundant_actions += 1
                info["redundant"] = True
            else:
                self.state.checked[vector] = True
                reward += USEFUL_ENUM_REWARD
                if self.host.is_viable(vector):
                    self.state.found[vector] = True
                    reward += DISCOVERY_REWARD
                    info["discovered_vector"] = vector

        elif action_enum == Action.ESCALATE:
            selected_vector = self._pick_lowest_risk_found_vector()
            if selected_vector is None:
                reward += PREMATURE_ESCALATION_PENALTY
                info["premature_escalation"] = True
            else:
                risk_penalty = VECTOR_RISK_PENALTIES[selected_vector]
                reward -= risk_penalty
                self._risk_exposure += risk_penalty
                self._escalation_attempts += 1

                success = attempt_escalation(self.host, selected_vector, self._rng)
                info["escalation_vector"] = selected_vector
                info["escalation_success"] = success

                if success:
                    self.state.current_privilege = 1
                    reward += SUCCESS_ESCALATION_REWARD
                else:
                    reward += FAILED_ESCALATION_PENALTY

        elif action_enum == Action.STOP:
            self._done = True

        self._steps += 1
        self._episode_reward += reward

        if self.state.current_privilege == 1:
            self._done = True
        if self._steps >= self.max_steps:
            self._done = True

        if self._done:
            info["episode"] = self._episode_metrics()

        return self.state.to_vector(), reward, self._done, info

