from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Sequence

import numpy as np

from environment import PrivilegeEscalationEnv


RawMetrics = Dict[str, np.ndarray]
Summary = Dict[str, float]
SelectorFactory = Callable[[], Callable[[np.ndarray], int]]


def run_episode(
    env: PrivilegeEscalationEnv, action_selector: Callable[[np.ndarray], int]
) -> Dict[str, float]:
    state = env.reset()
    done = False
    info: Dict[str, object] = {}

    while not done:
        action = action_selector(state)
        state, _, done, info = env.step(action)

    episode = info.get("episode")
    if episode is None:
        raise RuntimeError("Environment did not return episode metrics on termination.")
    return episode


def evaluate_for_seed(
    action_selector: Callable[[np.ndarray], int],
    episodes: int,
    seed: int,
    max_steps: int,
) -> RawMetrics:
    env = PrivilegeEscalationEnv(seed=seed, max_steps=max_steps)
    raw: Dict[str, List[float]] = {
        "success": [],
        "steps": [],
        "reward": [],
        "risk_exposure": [],
        "redundant_actions": [],
        "escalation_attempts": [],
    }

    for _ in range(episodes):
        metrics = run_episode(env, action_selector)
        raw["success"].append(float(metrics["success"]))
        raw["steps"].append(float(metrics["steps"]))
        raw["reward"].append(float(metrics["reward"]))
        raw["risk_exposure"].append(float(metrics["risk_exposure"]))
        raw["redundant_actions"].append(float(metrics["redundant_actions"]))
        raw["escalation_attempts"].append(float(metrics["escalation_attempts"]))

    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


def summarize_raw_metrics(raw: RawMetrics) -> Summary:
    return {
        "success_rate": float(raw["success"].mean()),
        "avg_steps": float(raw["steps"].mean()),
        "avg_reward": float(raw["reward"].mean()),
        "avg_risk_exposure": float(raw["risk_exposure"].mean()),
        "avg_redundant_actions": float(raw["redundant_actions"].mean()),
        "avg_escalation_attempts": float(raw["escalation_attempts"].mean()),
        "episodes": float(raw["success"].shape[0]),
    }


def evaluate_across_seeds(
    selector_factory: SelectorFactory,
    seeds: Sequence[int],
    episodes_per_seed: int,
    max_steps: int = 20,
) -> Dict[str, object]:
    per_seed: List[Dict[str, object]] = []
    combined: Dict[str, List[np.ndarray]] = {
        "success": [],
        "steps": [],
        "reward": [],
        "risk_exposure": [],
        "redundant_actions": [],
        "escalation_attempts": [],
    }

    for seed in seeds:
        selector = selector_factory()
        raw = evaluate_for_seed(
            action_selector=selector,
            episodes=episodes_per_seed,
            seed=seed,
            max_steps=max_steps,
        )
        summary = summarize_raw_metrics(raw)
        summary["seed"] = float(seed)
        per_seed.append(summary)

        for key in combined:
            combined[key].append(raw[key])

    merged_raw = {key: np.concatenate(chunks) for key, chunks in combined.items()}
    overall = summarize_raw_metrics(merged_raw)
    return {"per_seed": per_seed, "overall": overall, "raw": merged_raw}


def bootstrap_mean_difference(
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed: int = 0,
    samples: int = 5_000,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(samples, dtype=np.float32)

    baseline_size = baseline.shape[0]
    candidate_size = candidate.shape[0]
    for idx in range(samples):
        baseline_draw = rng.choice(baseline, size=baseline_size, replace=True)
        candidate_draw = rng.choice(candidate, size=candidate_size, replace=True)
        diffs[idx] = candidate_draw.mean() - baseline_draw.mean()

    return {
        "difference": float(candidate.mean() - baseline.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


def statistical_comparison(
    baseline_raw: Mapping[str, np.ndarray],
    candidate_raw: Mapping[str, np.ndarray],
    seed: int = 0,
    samples: int = 5_000,
) -> Dict[str, Dict[str, float]]:
    mapping = {
        "success_rate": "success",
        "avg_steps": "steps",
        "avg_reward": "reward",
        "avg_risk_exposure": "risk_exposure",
        "avg_redundant_actions": "redundant_actions",
        "avg_escalation_attempts": "escalation_attempts",
    }
    comparison: Dict[str, Dict[str, float]] = {}

    for idx, (output_key, raw_key) in enumerate(mapping.items()):
        stats = bootstrap_mean_difference(
            baseline=baseline_raw[raw_key],
            candidate=candidate_raw[raw_key],
            seed=seed + idx * 101,
            samples=samples,
        )
        comparison[output_key] = {
            "baseline_mean": float(baseline_raw[raw_key].mean()),
            "candidate_mean": float(candidate_raw[raw_key].mean()),
            "difference": stats["difference"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
        }
    return comparison
