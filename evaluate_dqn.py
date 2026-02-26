from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence

from baseline_policy import DeterministicBaselinePolicy
from dqn_agent import DQNAgent, DQNConfig
from evaluation import evaluate_across_seeds, statistical_comparison


def parse_seeds(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def evaluate_trained_agent(
    model_path: str,
    seeds: Sequence[int],
    episodes_per_seed: int = 1_000,
    max_steps: int = 20,
    device: Optional[str] = None,
    bootstrap_samples: int = 5_000,
) -> Dict[str, object]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    baseline_policy = DeterministicBaselinePolicy()
    baseline_results = evaluate_across_seeds(
        selector_factory=lambda: baseline_policy.select_action,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        max_steps=max_steps,
    )

    agent = DQNAgent(config=DQNConfig(), seed=0, device=device)
    agent.load(model_path)
    dqn_results = evaluate_across_seeds(
        selector_factory=lambda: (lambda state: agent.select_action(state, explore=False)),
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        max_steps=max_steps,
    )

    comparison = statistical_comparison(
        baseline_raw=baseline_results["raw"],
        candidate_raw=dqn_results["raw"],
        seed=123,
        samples=bootstrap_samples,
    )

    baseline_steps = float(baseline_results["overall"]["avg_steps"])
    dqn_steps = float(dqn_results["overall"]["avg_steps"])
    step_reduction = (baseline_steps - dqn_steps) / baseline_steps if baseline_steps > 0 else 0.0
    risk_diff_ci_low = float(comparison["avg_risk_exposure"]["ci_low"])

    criteria = {
        "step_reduction_percent": 100.0 * step_reduction,
        "dqn_success_rate": float(dqn_results["overall"]["success_rate"]),
        "dqn_avg_steps": dqn_steps,
        "meets_step_target_<=8": dqn_steps <= 8.0,
        "meets_success_target_>=82pct": dqn_results["overall"]["success_rate"] >= 0.82,
        "meets_20pct_step_reduction": step_reduction >= 0.20,
        "risk_not_significantly_increased": risk_diff_ci_low <= 0.0,
    }

    return {
        "baseline": baseline_results,
        "dqn": dqn_results,
        "comparison": comparison,
        "criteria": criteria,
    }


def print_report(report: Dict[str, object]) -> None:
    baseline = report["baseline"]["overall"]
    dqn = report["dqn"]["overall"]
    comparison = report["comparison"]
    criteria = report["criteria"]

    print("=== Baseline Overall ===")
    print(
        f"success={baseline['success_rate']:.4f} "
        f"steps={baseline['avg_steps']:.4f} "
        f"reward={baseline['avg_reward']:.4f} "
        f"risk={baseline['avg_risk_exposure']:.4f} "
        f"redundant={baseline['avg_redundant_actions']:.4f}"
    )

    print("=== DQN Overall ===")
    print(
        f"success={dqn['success_rate']:.4f} "
        f"steps={dqn['avg_steps']:.4f} "
        f"reward={dqn['avg_reward']:.4f} "
        f"risk={dqn['avg_risk_exposure']:.4f} "
        f"redundant={dqn['avg_redundant_actions']:.4f}"
    )

    print("=== Statistical Comparison (DQN - Baseline, 95% bootstrap CI) ===")
    for key, values in comparison.items():
        print(
            f"{key}: diff={values['difference']:.5f}, "
            f"ci=[{values['ci_low']:.5f}, {values['ci_high']:.5f}]"
        )

    print("=== Success Criteria ===")
    for key, value in criteria.items():
        print(f"{key}: {value}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN and compare to baseline.")
    parser.add_argument("--model-path", type=str, default="artifacts/dqn_model.pt")
    parser.add_argument("--seeds", type=str, default="7,42,1337")
    parser.add_argument("--episodes-per-seed", type=int, default=1_000)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    report = evaluate_trained_agent(
        model_path=args.model_path,
        seeds=parse_seeds(args.seeds),
        episodes_per_seed=args.episodes_per_seed,
        max_steps=args.max_steps,
        device=args.device,
        bootstrap_samples=args.bootstrap_samples,
    )
    print_report(report)


if __name__ == "__main__":
    main()

