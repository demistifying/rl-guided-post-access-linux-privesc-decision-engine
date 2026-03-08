"""
Evaluate a trained DQN against the deterministic baseline policy.
Runs both policies across multiple seeds and computes bootstrap confidence intervals.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from postex_agent.environment.simulation_env import SimulationEnv
from postex_agent.rl.baseline_policy import BaselinePolicy


RawMetrics = Dict[str, np.ndarray]
Summary = Dict[str, float]
SelectorFn = Callable[[np.ndarray], int]
SelectorFactory = Callable[[], SelectorFn]


def run_episode(env: SimulationEnv, selector: SelectorFn) -> Dict[str, float]:
    state = env.reset()
    done = False
    info: dict = {}
    while not done:
        action = selector(state)
        state, _, done, info = env.step(action)
    ep = info.get("episode")
    if ep is None:
        raise RuntimeError("Environment did not return episode metrics.")
    return ep


def evaluate_seed(selector: SelectorFn, seed: int, episodes: int, max_steps: int) -> RawMetrics:
    env = SimulationEnv(seed=seed, max_steps=max_steps)
    raw: Dict[str, List[float]] = {
        key: []
        for key in [
            "success",
            "steps",
            "reward",
            "risk_exposure",
            "redundant_actions",
            "escalation_attempts",
        ]
    }
    for _ in range(episodes):
        ep = run_episode(env, selector)
        for key in raw:
            raw[key].append(float(ep.get(key, 0)))
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


def summarize(raw: RawMetrics) -> Summary:
    return {
        "success_rate": float(raw["success"].mean()),
        "avg_steps": float(raw["steps"].mean()),
        "avg_reward": float(raw["reward"].mean()),
        "avg_risk_exposure": float(raw["risk_exposure"].mean()),
        "avg_redundant_actions": float(raw["redundant_actions"].mean()),
        "avg_escalation_attempts": float(raw["escalation_attempts"].mean()),
        "n_episodes": float(raw["success"].shape[0]),
    }


def evaluate_across_seeds(
    factory: SelectorFactory,
    seeds: Sequence[int],
    episodes_per_seed: int,
    max_steps: int,
) -> Dict[str, object]:
    per_seed: List[Summary] = []
    combined: Dict[str, List[np.ndarray]] = {
        key: []
        for key in [
            "success",
            "steps",
            "reward",
            "risk_exposure",
            "redundant_actions",
            "escalation_attempts",
        ]
    }

    for seed in seeds:
        selector = factory()
        raw = evaluate_seed(selector, seed, episodes_per_seed, max_steps)
        seed_summary = summarize(raw)
        seed_summary["seed"] = float(seed)
        per_seed.append(seed_summary)
        for key in combined:
            combined[key].append(raw[key])

    merged = {k: np.concatenate(v) for k, v in combined.items()}
    return {"per_seed": per_seed, "overall": summarize(merged), "raw": merged}


def bootstrap_diff(
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed: int = 0,
    samples: int = 5000,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(samples, dtype=np.float32)
    n_base, n_cand = baseline.shape[0], candidate.shape[0]
    for idx in range(samples):
        diffs[idx] = (
            rng.choice(candidate, size=n_cand, replace=True).mean()
            - rng.choice(baseline, size=n_base, replace=True).mean()
        )
    return {
        "difference": float(candidate.mean() - baseline.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
    }


def statistical_comparison(
    baseline_raw: Mapping[str, np.ndarray],
    candidate_raw: Mapping[str, np.ndarray],
    seed: int = 0,
    samples: int = 5000,
) -> Dict[str, Dict[str, float]]:
    metrics = {
        "success_rate": "success",
        "avg_steps": "steps",
        "avg_reward": "reward",
        "avg_risk_exposure": "risk_exposure",
        "avg_redundant_actions": "redundant_actions",
        "avg_escalation_attempts": "escalation_attempts",
    }
    return {
        label: bootstrap_diff(
            baseline_raw[key],
            candidate_raw[key],
            seed=seed + idx * 101,
            samples=samples,
        )
        for idx, (label, key) in enumerate(metrics.items())
    }


def success_criteria(
    baseline_overall: Summary,
    dqn_overall: Summary,
    comparison: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    base_steps = baseline_overall["avg_steps"]
    dqn_steps = dqn_overall["avg_steps"]
    step_reduction = (base_steps - dqn_steps) / base_steps if base_steps > 0 else 0.0
    return {
        "dqn_success_rate": dqn_overall["success_rate"],
        "baseline_success_rate": baseline_overall["success_rate"],
        "dqn_avg_steps": dqn_steps,
        "baseline_avg_steps": base_steps,
        "step_reduction_pct": round(100.0 * step_reduction, 2),
        "meets_success_target_>=82pct": dqn_overall["success_rate"] >= 0.82,
        "meets_step_target_<=8": dqn_steps <= 8.0,
        "meets_20pct_step_reduction": step_reduction >= 0.20,
        "risk_not_significantly_worse": comparison["avg_risk_exposure"]["ci_low"] <= 0.0,
    }


def print_report(report: Dict[str, object]) -> None:
    b = report["baseline"]["overall"]
    d = report["dqn"]["overall"]
    c = report["comparison"]
    criteria = report["criteria"]

    width = 72
    print("\n" + "=" * width)
    print("EVALUATION REPORT - DQN vs Deterministic Baseline")
    print("=" * width)

    print(f"\n{'Metric':<30} {'Baseline':>10} {'DQN':>10} {'Delta':>10}")
    print("-" * width)
    rows = [
        ("Success rate", "success_rate", True),
        ("Avg steps", "avg_steps", False),
        ("Avg reward", "avg_reward", True),
        ("Avg risk exposure", "avg_risk_exposure", False),
        ("Avg redundant actions", "avg_redundant_actions", False),
        ("Avg escalation attempts", "avg_escalation_attempts", False),
    ]
    for label, key, higher_better in rows:
        baseline_value = b[key]
        dqn_value = d[key]
        delta = dqn_value - baseline_value
        direction = "UP" if (delta > 0) == higher_better else "DOWN"
        print(f"  {label:<28} {baseline_value:>10.4f} {dqn_value:>10.4f} {delta:>+9.4f} {direction}")

    print(f"\n{'-' * width}")
    print("Bootstrap 95% CI (DQN - Baseline)")
    print(f"  {'Metric':<28} {'Diff':>10}  {'CI':>24}")
    print(f"  {'-' * 66}")
    for label, stats in c.items():
        diff = stats["difference"]
        low = stats["ci_low"]
        high = stats["ci_high"]
        print(f"  {label:<28} {diff:>+10.4f}  [{low:>+9.4f}, {high:>+9.4f}]")

    print(f"\n{'-' * width}")
    print("Success Criteria")
    print(f"  {'-' * 66}")
    for key, value in criteria.items():
        if isinstance(value, bool):
            status = "[OK]" if value else "[NO]"
            print(f"  {status} {key}: {value}")
        else:
            print(f"  [..] {key}: {value}")
    print("=" * width + "\n")


def evaluate(
    model_path: str,
    seeds: Sequence[int],
    episodes_per_seed: int = 1000,
    max_steps: int = 20,
    bootstrap_samples: int = 5000,
    device: Optional[str] = None,
    report_path: Optional[str] = None,
) -> Dict[str, object]:
    print("[eval] Running baseline policy...")
    baseline_policy = BaselinePolicy()
    baseline_results = evaluate_across_seeds(
        factory=lambda: baseline_policy.select_action,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        max_steps=max_steps,
    )

    print(f"[eval] Loading DQN from {model_path}...")
    try:
        from postex_agent.rl.policy_inference import RLPolicy

        rl_policy = RLPolicy(model_path=model_path, device=device)
        dqn_results = evaluate_across_seeds(
            factory=lambda: (lambda sv: int(rl_policy.predict(sv))),
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            max_steps=max_steps,
        )
    except (FileNotFoundError, ImportError, ModuleNotFoundError) as exc:
        print(f"[eval] WARNING: {exc}")
        print("[eval] Falling back to baseline-only results.")
        dqn_results = baseline_results

    print("[eval] Computing bootstrap statistics...")
    comparison = statistical_comparison(
        baseline_raw=baseline_results["raw"],
        candidate_raw=dqn_results["raw"],
        seed=123,
        samples=bootstrap_samples,
    )

    criteria = success_criteria(
        baseline_overall=baseline_results["overall"],
        dqn_overall=dqn_results["overall"],
        comparison=comparison,
    )

    report = {
        "baseline": baseline_results,
        "dqn": dqn_results,
        "comparison": comparison,
        "criteria": criteria,
        "config": {
            "model_path": model_path,
            "seeds": list(seeds),
            "episodes_per_seed": episodes_per_seed,
            "max_steps": max_steps,
            "bootstrap_samples": bootstrap_samples,
        },
    }

    save_path = report_path or "artifacts/eval_report.json"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    serializable = {
        "baseline": {
            "overall": baseline_results["overall"],
            "per_seed": baseline_results["per_seed"],
        },
        "dqn": {
            "overall": dqn_results["overall"],
            "per_seed": dqn_results["per_seed"],
        },
        "comparison": comparison,
        "criteria": criteria,
        "config": report["config"],
    }
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, default=float)
    print(f"[eval] Report saved to {save_path}")

    return report


def parse_seeds(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained DQN vs deterministic baseline on the privilege escalation MDP."
    )
    parser.add_argument("--model-path", default="artifacts/dqn_model.pt")
    parser.add_argument("--seeds", default="7,42,1337")
    parser.add_argument("--episodes-per-seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--report-path", default="artifacts/eval_report.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    report = evaluate(
        model_path=args.model_path,
        seeds=parse_seeds(args.seeds),
        episodes_per_seed=args.episodes_per_seed,
        max_steps=args.max_steps,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
        report_path=args.report_path,
    )
    print_report(report)


if __name__ == "__main__":
    _cli()

