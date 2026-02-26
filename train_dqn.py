from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dqn_agent import DQNAgent, DQNConfig
from environment import PrivilegeEscalationEnv


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rolling_mean(history: List[Dict[str, float]], key: str) -> float:
    values = [item[key] for item in history]
    if not values:
        return 0.0
    return float(np.mean(values))


def _save_history_csv(path: str, rows: List[Dict[str, float]]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = [
        "episode",
        "epsilon",
        "reward",
        "steps",
        "success",
        "risk_exposure",
        "redundant_actions",
        "escalation_attempts",
        "mean_loss",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_agent(
    episodes: int = 10_000,
    seed: int = 42,
    max_steps: int = 20,
    model_path: str = "artifacts/dqn_model.pt",
    history_path: str = "artifacts/dqn_training_history.csv",
    log_interval: int = 500,
    device: Optional[str] = None,
) -> Tuple[DQNAgent, Dict[str, float]]:
    set_global_seed(seed)
    env = PrivilegeEscalationEnv(seed=seed, max_steps=max_steps)
    config = DQNConfig()
    agent = DQNAgent(config=config, seed=seed, device=device)

    total_env_steps = 0
    history: List[Dict[str, float]] = []

    for episode_idx in range(episodes):
        state = env.reset()
        agent.set_epsilon_for_episode(episode_idx)

        done = False
        losses: List[float] = []
        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_env_steps += 1
            if total_env_steps % config.target_update_interval == 0:
                agent.hard_update_target()

        episode_metrics = info["episode"]
        row = {
            "episode": float(episode_idx + 1),
            "epsilon": float(agent.epsilon),
            "reward": float(episode_metrics["reward"]),
            "steps": float(episode_metrics["steps"]),
            "success": float(episode_metrics["success"]),
            "risk_exposure": float(episode_metrics["risk_exposure"]),
            "redundant_actions": float(episode_metrics["redundant_actions"]),
            "escalation_attempts": float(episode_metrics["escalation_attempts"]),
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
        }
        history.append(row)

        if (episode_idx + 1) % log_interval == 0:
            window = history[-log_interval:]
            print(
                f"[train] episode={episode_idx + 1} "
                f"epsilon={agent.epsilon:.3f} "
                f"success={_rolling_mean(window, 'success'):.3f} "
                f"steps={_rolling_mean(window, 'steps'):.3f} "
                f"reward={_rolling_mean(window, 'reward'):.3f}"
            )

    agent.save(model_path)
    _save_history_csv(history_path, history)

    summary = {
        "episodes": float(episodes),
        "final_epsilon": float(agent.epsilon),
        "mean_success": _rolling_mean(history, "success"),
        "mean_steps": _rolling_mean(history, "steps"),
        "mean_reward": _rolling_mean(history, "reward"),
        "mean_risk_exposure": _rolling_mean(history, "risk_exposure"),
        "model_path": model_path,
        "history_path": history_path,
    }
    return agent, summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DQN on privilege escalation MDP.")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--model-path", type=str, default="artifacts/dqn_model.pt")
    parser.add_argument(
        "--history-path", type=str, default="artifacts/dqn_training_history.csv"
    )
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _, summary = train_agent(
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        model_path=args.model_path,
        history_path=args.history_path,
        log_interval=args.log_interval,
        device=args.device,
    )
    print("[train] complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
