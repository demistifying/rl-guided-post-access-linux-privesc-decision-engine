"""
DQN training entrypoint.
Trains the agent on the SimulationEnv and saves the model + history CSV.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from postex_agent.environment.simulation_env import SimulationEnv
from postex_agent.rl.dqn_agent import DQNAgent, DQNConfig


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rolling_mean(history: List[Dict[str, float]], key: str, window: int) -> float:
    vals = [h[key] for h in history[-window:]]
    return float(np.mean(vals)) if vals else 0.0


def _save_csv(path: str, rows: List[Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train(
    episodes:      int = 10_000,
    seed:          int = 42,
    max_steps:     int = 20,
    model_path:    str = "artifacts/dqn_model.pt",
    history_path:  str = "artifacts/dqn_training_history.csv",
    log_interval:  int = 500,
    device:        Optional[str] = None,
) -> Tuple[DQNAgent, Dict[str, float]]:

    set_global_seed(seed)
    env    = SimulationEnv(seed=seed, max_steps=max_steps)
    config = DQNConfig()
    agent  = DQNAgent(config=config, seed=seed, device=device)

    total_steps = 0
    history: List[Dict[str, float]] = []

    for ep in range(episodes):
        state = env.reset()
        agent.set_epsilon(ep)
        done   = False
        losses: List[float] = []

        while not done:
            action     = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            state        = next_state
            total_steps += 1
            if total_steps % config.target_update_interval == 0:
                agent.hard_update_target()

        ep_info = info.get("episode", {})
        row = {
            "episode":             float(ep + 1),
            "epsilon":             float(agent.epsilon),
            "reward":              float(ep_info.get("reward", 0)),
            "steps":               float(ep_info.get("steps", 0)),
            "success":             float(ep_info.get("success", False)),
            "risk_exposure":       float(ep_info.get("risk_exposure", 0)),
            "redundant_actions":   float(ep_info.get("redundant_actions", 0)),
            "escalation_attempts": float(ep_info.get("escalation_attempts", 0)),
            "mean_loss":           float(np.mean(losses)) if losses else 0.0,
        }
        history.append(row)

        if (ep + 1) % log_interval == 0:
            w = log_interval
            print(
                f"[train] ep={ep+1:>6d} "
                f"eps={agent.epsilon:.3f} "
                f"success={_rolling_mean(history, 'success', w):.3f} "
                f"steps={_rolling_mean(history, 'steps', w):.2f} "
                f"reward={_rolling_mean(history, 'reward', w):.3f}"
            )

    agent.save(model_path)
    _save_csv(history_path, history)

    summary: Dict[str, float] = {
        "episodes":         float(episodes),
        "final_epsilon":    float(agent.epsilon),
        "mean_success":     _rolling_mean(history, "success", episodes),
        "mean_steps":       _rolling_mean(history, "steps", episodes),
        "mean_reward":      _rolling_mean(history, "reward", episodes),
        "mean_risk":        _rolling_mean(history, "risk_exposure", episodes),
    }
    return agent, summary


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train DQN agent on privilege escalation MDP.")
    parser.add_argument("--episodes",     type=int, default=10_000)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--max-steps",    type=int, default=20)
    parser.add_argument("--model-path",   type=str, default="artifacts/dqn_model.pt")
    parser.add_argument("--history-path", type=str, default="artifacts/dqn_training_history.csv")
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--device",       type=str, default=None)
    args = parser.parse_args()

    _, summary = train(
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        model_path=args.model_path,
        history_path=args.history_path,
        log_interval=args.log_interval,
        device=args.device,
    )
    print("[train] complete")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()

