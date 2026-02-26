from __future__ import annotations

import argparse
from typing import List

from evaluate_dqn import evaluate_trained_agent, parse_seeds, print_report
from train_dqn import train_agent


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline vs DQN experiment for privilege escalation MDP."
    )
    parser.add_argument("--seeds", type=str, default="7,42,1337")
    parser.add_argument("--episodes-per-seed", type=int, default=1_000)
    parser.add_argument("--train-episodes", type=int, default=10_000)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--model-path", type=str, default="artifacts/dqn_model.pt")
    parser.add_argument(
        "--history-path", type=str, default="artifacts/dqn_training_history.csv"
    )
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    seeds: List[int] = parse_seeds(args.seeds)

    if not args.skip_training:
        print("=== Training DQN ===")
        _, summary = train_agent(
            episodes=args.train_episodes,
            seed=args.train_seed,
            max_steps=args.max_steps,
            model_path=args.model_path,
            history_path=args.history_path,
            log_interval=args.log_interval,
            device=args.device,
        )
        print("=== Training Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

    print("=== Evaluation ===")
    report = evaluate_trained_agent(
        model_path=args.model_path,
        seeds=seeds,
        episodes_per_seed=args.episodes_per_seed,
        max_steps=args.max_steps,
        device=args.device,
        bootstrap_samples=args.bootstrap_samples,
    )
    print_report(report)


if __name__ == "__main__":
    main()

