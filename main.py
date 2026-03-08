"""
One-shot train + evaluate pipeline.

Usage:
    python main.py
    python main.py --skip-training --model-path artifacts/dqn_model.pt
    python main.py --train-episodes 5000 --seeds 7,42,1337 --episodes-per-seed 500
"""
from __future__ import annotations

import argparse
from typing import List

from postex_agent.rl.train_dqn import train
from postex_agent.rl.evaluate_dqn import evaluate, parse_seeds, print_report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and evaluate the postex_agent DQN.")
    p.add_argument("--train-episodes",    type=int,   default=10_000)
    p.add_argument("--train-seed",        type=int,   default=42)
    p.add_argument("--max-steps",         type=int,   default=20)
    p.add_argument("--model-path",        type=str,   default="artifacts/dqn_model.pt")
    p.add_argument("--history-path",      type=str,   default="artifacts/dqn_training_history.csv")
    p.add_argument("--log-interval",      type=int,   default=500)
    p.add_argument("--seeds",             type=str,   default="7,42,1337")
    p.add_argument("--episodes-per-seed", type=int,   default=1_000)
    p.add_argument("--bootstrap-samples", type=int,   default=5_000)
    p.add_argument("--report-path",       type=str,   default="artifacts/eval_report.json")
    p.add_argument("--device",            type=str,   default=None)
    p.add_argument("--skip-training",     action="store_true")
    return p


def main() -> None:
    args  = _build_parser().parse_args()
    seeds: List[int] = parse_seeds(args.seeds)

    if not args.skip_training:
        print("=" * 60)
        print("Training DQN")
        print("=" * 60)
        _, summary = train(
            episodes=args.train_episodes,
            seed=args.train_seed,
            max_steps=args.max_steps,
            model_path=args.model_path,
            history_path=args.history_path,
            log_interval=args.log_interval,
            device=args.device,
        )
        print("\nTraining summary:")
        for k, v in summary.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Evaluating")
    print("=" * 60)
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
