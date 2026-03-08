from __future__ import annotations

import argparse
from typing import List

from postex_agent.rl.evaluate_dqn import evaluate, parse_seeds, print_report
from postex_agent.rl.train_dqn import train


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate postex_agent DQN.")
    parser.add_argument("--seeds", type=str, default="7,42,1337")
    parser.add_argument("--episodes-per-seed", type=int, default=1_000)
    parser.add_argument("--train-episodes", type=int, default=10_000)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--model-path", type=str, default="artifacts/dqn_model.pt")
    parser.add_argument("--history-path", type=str, default="artifacts/dqn_training_history.csv")
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--report-path", type=str, default="artifacts/eval_report.json")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    seeds: List[int] = parse_seeds(args.seeds)

    if not args.skip_training:
        print("=== Training DQN ===")
        _, summary = train(
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
    report = evaluate(
        model_path=args.model_path,
        seeds=seeds,
        episodes_per_seed=args.episodes_per_seed,
        max_steps=args.max_steps,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
        report_path=args.report_path,
    )
    print_report(report)


if __name__ == "__main__":
    main()

