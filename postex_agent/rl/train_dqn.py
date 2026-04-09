"""
DQN training entrypoint.

Trains the agent on the SimulationEnv and saves:
  - final model checkpoint
  - best validation checkpoint
  - per-episode training CSV
  - per-interval validation CSV
  - human-readable training log
  - training curve PNG
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from postex_agent.environment.simulation_env import SimulationEnv
from postex_agent.rl.dqn_agent import DQNAgent, DQNConfig


ValidationMetrics = Dict[str, float]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rolling_mean(history: List[Dict[str, float]], key: str, window: int) -> float:
    values = [row[key] for row in history[-window:]]
    return float(np.mean(values)) if values else 0.0


def _save_csv(path: str, rows: List[Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _default_best_model_path(model_path: str) -> str:
    root, ext = os.path.splitext(model_path)
    return f"{root}_best{ext or '.pt'}"


def _validation_rank(metrics: ValidationMetrics) -> Tuple[float, float, float, float]:
    """Rank by success first, then faster / lower-risk behaviour."""
    return (
        float(metrics["success_rate"]),
        -float(metrics["avg_steps"]),
        -float(metrics["avg_risk_exposure"]),
        float(metrics["avg_reward"]),
    )


def _is_better_validation(
    candidate: ValidationMetrics,
    incumbent: Optional[ValidationMetrics],
) -> bool:
    if incumbent is None:
        return True
    return _validation_rank(candidate) > _validation_rank(incumbent)


def evaluate_validation(
    agent: DQNAgent,
    *,
    episodes: int,
    seed: int,
    max_steps: int,
) -> ValidationMetrics:
    """Run deterministic validation episodes for checkpoint selection."""
    env = SimulationEnv(seed=seed, max_steps=max_steps)

    total_success = 0
    total_steps = 0.0
    total_reward = 0.0
    total_risk = 0.0

    for episode_index in range(episodes):
        state = env.reset(seed=seed + episode_index)
        mask = env.action_mask()
        done = False
        info: Dict[str, object] = {}

        while not done:
            action = agent.select_action(state, explore=False, mask=mask)
            state, _, done, info = env.step(action)
            mask = info.get("action_mask", mask)

        episode = info.get("episode", {})
        if isinstance(episode, dict):
            total_success += int(bool(episode.get("success", False)))
            total_steps += float(episode.get("steps", 0.0))
            total_reward += float(episode.get("reward", 0.0))
            total_risk += float(episode.get("risk_exposure", 0.0))

    denom = float(max(episodes, 1))
    return {
        "success_rate": total_success / denom,
        "avg_steps": total_steps / denom,
        "avg_reward": total_reward / denom,
        "avg_risk_exposure": total_risk / denom,
    }


def _write_training_log(
    log_path: str,
    config: DQNConfig,
    episodes: int,
    seed: int,
    max_steps: int,
    device: str,
    history: List[Dict[str, float]],
    summary: Dict[str, float],
    interval_logs: List[str],
    elapsed_sec: float,
    model_path: str,
    history_path: str,
    validation_history: Optional[List[Dict[str, float]]] = None,
    best_validation: Optional[ValidationMetrics] = None,
    best_validation_episode: Optional[int] = None,
    best_model_path: Optional[str] = None,
    validation_history_path: Optional[str] = None,
) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    milestones: List[str] = []
    for window in (100, 500, 1000):
        if len(history) >= window:
            milestones.append(
                f"  Last {window:>5d} eps:  "
                f"success={_rolling_mean(history, 'success', window):.4f}  "
                f"reward={_rolling_mean(history, 'reward', window):>8.3f}  "
                f"steps={_rolling_mean(history, 'steps', window):>6.2f}  "
                f"loss={_rolling_mean(history, 'mean_loss', window):>7.4f}  "
                f"risk={_rolling_mean(history, 'risk_exposure', window):>6.4f}"
            )

    best_episode = max(history, key=lambda row: row["reward"])
    worst_episode = min(history, key=lambda row: row["reward"])

    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("=" * 72 + "\n")
        handle.write("  PostEx Agent - DQN Training Log\n")
        handle.write("=" * 72 + "\n\n")
        handle.write(f"  Timestamp      : {timestamp}\n")
        handle.write(f"  Duration       : {elapsed_sec:.1f}s ({elapsed_sec / 60:.1f}m)\n")
        handle.write(f"  Device         : {device}\n\n")

        handle.write("-" * 72 + "\n")
        handle.write("  HYPERPARAMETERS\n")
        handle.write("-" * 72 + "\n")
        handle.write(f"  Episodes              : {episodes}\n")
        handle.write(f"  Seed                  : {seed}\n")
        handle.write(f"  Max steps / episode   : {max_steps}\n")
        handle.write(
            f"  Network               : "
            f"{config.state_dim}->{config.hidden_dim}->{config.hidden_dim}->{config.action_dim}\n"
        )
        handle.write(f"  Gamma                 : {config.gamma}\n")
        handle.write(f"  Learning rate         : {config.learning_rate}\n")
        handle.write(f"  Replay buffer size    : {config.replay_buffer_size}\n")
        handle.write(f"  Batch size            : {config.batch_size}\n")
        handle.write(f"  Target net update     : every {config.target_update_interval} steps\n")
        handle.write(
            f"  Epsilon schedule      : "
            f"{config.epsilon_start}->{config.epsilon_end} over "
            f"{config.epsilon_decay_episodes} eps\n\n"
        )

        handle.write("-" * 72 + "\n")
        handle.write("  TRAINING PROGRESS\n")
        handle.write("-" * 72 + "\n")
        handle.write(
            f"  {'Episode':>8s}  {'Epsilon':>8s}  {'Success':>8s}  {'Reward':>8s}  "
            f"{'Steps':>8s}  {'Loss':>10s}  {'Risk':>8s}  {'Redundant':>10s}\n"
        )
        handle.write(
            f"  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  "
            f"{'--------':>8s}  {'----------':>10s}  {'--------':>8s}  {'----------':>10s}\n"
        )
        for line in interval_logs:
            handle.write(f"  {line}\n")
        handle.write("\n")

        handle.write("-" * 72 + "\n")
        handle.write("  ROLLING-WINDOW SUMMARY\n")
        handle.write("-" * 72 + "\n")
        for line in milestones:
            handle.write(line + "\n")
        handle.write("\n")

        if validation_history:
            handle.write("-" * 72 + "\n")
            handle.write("  VALIDATION SUMMARY\n")
            handle.write("-" * 72 + "\n")
            for row in validation_history:
                handle.write(
                    "  "
                    f"ep={int(row['episode']):>6d}  "
                    f"success={row['success_rate']:.4f}  "
                    f"steps={row['avg_steps']:.2f}  "
                    f"reward={row['avg_reward']:.3f}  "
                    f"risk={row['avg_risk_exposure']:.4f}\n"
                )
            if best_validation is not None and best_validation_episode is not None:
                handle.write("\n")
                handle.write(
                    "  Best validation checkpoint: "
                    f"ep {best_validation_episode}  "
                    f"success={best_validation['success_rate']:.4f}  "
                    f"steps={best_validation['avg_steps']:.2f}  "
                    f"risk={best_validation['avg_risk_exposure']:.4f}\n"
                )
            handle.write("\n")

        handle.write("-" * 72 + "\n")
        handle.write("  FINAL RESULTS\n")
        handle.write("-" * 72 + "\n")
        for key, value in summary.items():
            if isinstance(value, float):
                handle.write(f"  {key:<24s}: {value:.4f}\n")
            else:
                handle.write(f"  {key:<24s}: {value}\n")
        handle.write("\n")

        handle.write("-" * 72 + "\n")
        handle.write("  NOTABLE EPISODES\n")
        handle.write("-" * 72 + "\n")
        handle.write(
            f"  Best  reward : ep {int(best_episode['episode']):>6d}  "
            f"reward={best_episode['reward']:>8.2f}  "
            f"steps={int(best_episode['steps']):>3d}  "
            f"success={int(best_episode['success'])}\n"
        )
        handle.write(
            f"  Worst reward : ep {int(worst_episode['episode']):>6d}  "
            f"reward={worst_episode['reward']:>8.2f}  "
            f"steps={int(worst_episode['steps']):>3d}  "
            f"success={int(worst_episode['success'])}\n\n"
        )

        handle.write("-" * 72 + "\n")
        handle.write("  OUTPUT FILES\n")
        handle.write("-" * 72 + "\n")
        handle.write(f"  Final checkpoint: {model_path}\n")
        if best_model_path:
            handle.write(f"  Best checkpoint : {best_model_path}\n")
        handle.write(f"  Training CSV    : {history_path}\n")
        if validation_history_path:
            handle.write(f"  Validation CSV  : {validation_history_path}\n")
        handle.write(f"  Training log    : {log_path}\n")
        curve_path = os.path.join(os.path.dirname(log_path), "training_curve.png")
        handle.write(f"  Training curve  : {curve_path}\n\n")
        handle.write("=" * 72 + "\n")


def _plot_training_curves(history: List[Dict[str, float]], output_path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[train] matplotlib not available - skipping curve plot")
        return

    episodes = [int(row["episode"]) for row in history]
    window = min(500, len(history) // 4) or 1

    def _rolling(key: str) -> List[float]:
        values = [row[key] for row in history]
        rolled: List[float] = []
        for index in range(len(values)):
            start = max(0, index - window + 1)
            rolled.append(float(np.mean(values[start : index + 1])))
        return rolled

    rolling_success = _rolling("success")
    rolling_reward = _rolling("reward")
    rolling_steps = _rolling("steps")
    rolling_loss = _rolling("mean_loss")
    epsilons = [row["epsilon"] for row in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"DQN Training Curves ({len(history):,} Episodes)",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(episodes, rolling_success, color="#2ecc71", linewidth=1.5)
    ax.axhline(y=0.85, color="#e74c3c", linestyle="--", alpha=0.7, label="Target (0.85)")
    ax.set_title(f"Success Rate ({window}-ep rolling)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(episodes, rolling_reward, color="#3498db", linewidth=1.5)
    ax.set_title(f"Mean Reward ({window}-ep rolling)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(episodes, rolling_steps, color="#e67e22", linewidth=1.5)
    ax.set_title(f"Mean Steps per Episode ({window}-ep rolling)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(episodes, rolling_loss, color="#9b59b6", linewidth=1.5, label="Loss")
    ax2 = ax.twinx()
    ax2.plot(episodes, epsilons, color="#95a5a6", linewidth=1, alpha=0.6, label="Epsilon")
    ax2.set_ylabel("Epsilon", color="#95a5a6")
    ax.set_title("Mean Loss and Epsilon Schedule")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] saved training curve -> {output_path}")


def train(
    episodes: int = 10_000,
    seed: int = 42,
    max_steps: int = 20,
    model_path: str = "artifacts/dqn_model.pt",
    history_path: str = "artifacts/dqn_training_history.csv",
    log_interval: int = 500,
    validation_episodes: int = 200,
    validation_interval: int = 500,
    validation_seed: Optional[int] = None,
    best_model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[DQNAgent, Dict[str, float]]:
    set_global_seed(seed)
    env = SimulationEnv(seed=seed, max_steps=max_steps)
    config = DQNConfig()
    agent = DQNAgent(config=config, seed=seed, device=device)
    resolved_device = str(agent.device)

    validation_seed = validation_seed if validation_seed is not None else seed + 10_000
    validation_interval = validation_interval or log_interval
    best_model_path = best_model_path or _default_best_model_path(model_path)
    validation_enabled = validation_episodes > 0 and validation_interval > 0

    total_steps = 0
    history: List[Dict[str, float]] = []
    validation_history: List[Dict[str, float]] = []
    interval_logs: List[str] = []
    best_validation: Optional[ValidationMetrics] = None
    best_validation_episode: Optional[int] = None
    start_time = time.time()

    for episode_index in range(episodes):
        state = env.reset()
        mask = env.action_mask()
        agent.set_epsilon(episode_index)
        agent.set_beta(episode_index)
        done = False
        losses: List[float] = []

        while not done:
            action = agent.select_action(state, explore=True, mask=mask)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            state = next_state
            mask = info.get("action_mask", mask)
            total_steps += 1
            if total_steps % config.target_update_interval == 0:
                agent.hard_update_target()

        episode = info.get("episode", {})
        row = {
            "episode": float(episode_index + 1),
            "epsilon": float(agent.epsilon),
            "reward": float(episode.get("reward", 0.0)),
            "steps": float(episode.get("steps", 0.0)),
            "success": float(episode.get("success", False)),
            "risk_exposure": float(episode.get("risk_exposure", 0.0)),
            "redundant_actions": float(episode.get("redundant_actions", 0.0)),
            "escalation_attempts": float(episode.get("escalation_attempts", 0.0)),
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
        }
        history.append(row)

        if (episode_index + 1) % log_interval == 0:
            window = log_interval
            rolling_success = _rolling_mean(history, "success", window)
            rolling_reward = _rolling_mean(history, "reward", window)
            rolling_steps = _rolling_mean(history, "steps", window)
            rolling_loss = _rolling_mean(history, "mean_loss", window)
            rolling_risk = _rolling_mean(history, "risk_exposure", window)
            rolling_redundant = _rolling_mean(history, "redundant_actions", window)

            print(
                f"[train] ep={episode_index+1:>6d} "
                f"eps={agent.epsilon:.3f} "
                f"success={rolling_success:.3f} "
                f"steps={rolling_steps:.2f} "
                f"reward={rolling_reward:.3f}"
            )
            interval_logs.append(
                f"{episode_index+1:>8d}  "
                f"{agent.epsilon:>8.3f}  "
                f"{rolling_success:>8.4f}  "
                f"{rolling_reward:>8.3f}  "
                f"{rolling_steps:>8.2f}  "
                f"{rolling_loss:>10.4f}  "
                f"{rolling_risk:>8.4f}  "
                f"{rolling_redundant:>10.2f}"
            )

        if validation_enabled and (episode_index + 1) % validation_interval == 0:
            metrics = evaluate_validation(
                agent,
                episodes=validation_episodes,
                seed=validation_seed,
                max_steps=max_steps,
            )
            metrics["episode"] = float(episode_index + 1)
            validation_history.append(metrics)
            print(
                "[train] validation "
                f"ep={episode_index+1:>6d} "
                f"success={metrics['success_rate']:.3f} "
                f"steps={metrics['avg_steps']:.2f} "
                f"risk={metrics['avg_risk_exposure']:.3f}"
            )
            candidate = {key: value for key, value in metrics.items() if key != "episode"}
            if _is_better_validation(candidate, best_validation):
                best_validation = candidate
                best_validation_episode = episode_index + 1
                agent.save(best_model_path)
                print(f"[train] saved new best checkpoint -> {best_model_path}")

    elapsed = time.time() - start_time

    agent.save(model_path)
    _save_csv(history_path, history)

    artifacts_dir = os.path.dirname(model_path) or "artifacts"
    validation_history_path = os.path.join(artifacts_dir, "validation_history.csv")
    if validation_history:
        _save_csv(validation_history_path, validation_history)
    elif os.path.exists(validation_history_path):
        os.remove(validation_history_path)

    if not validation_enabled:
        best_model_path = model_path
    elif best_validation is None:
        agent.save(best_model_path)
        best_validation = evaluate_validation(
            agent,
            episodes=validation_episodes,
            seed=validation_seed,
            max_steps=max_steps,
        )
        best_validation_episode = episodes

    summary: Dict[str, float] = {
        "episodes": float(episodes),
        "final_epsilon": float(agent.epsilon),
        "mean_success": _rolling_mean(history, "success", episodes),
        "mean_steps": _rolling_mean(history, "steps", episodes),
        "mean_reward": _rolling_mean(history, "reward", episodes),
        "mean_risk": _rolling_mean(history, "risk_exposure", episodes),
    }
    if best_validation is not None:
        summary.update({
            "best_validation_success": best_validation["success_rate"],
            "best_validation_steps": best_validation["avg_steps"],
            "best_validation_reward": best_validation["avg_reward"],
            "best_validation_risk": best_validation["avg_risk_exposure"],
        })
    if best_validation_episode is not None:
        summary["best_validation_episode"] = float(best_validation_episode)

    log_path = os.path.join(artifacts_dir, "training_log.txt")
    curve_path = os.path.join(artifacts_dir, "training_curve.png")
    _write_training_log(
        log_path=log_path,
        config=config,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
        device=resolved_device,
        history=history,
        summary=summary,
        interval_logs=interval_logs,
        elapsed_sec=elapsed,
        model_path=model_path,
        history_path=history_path,
        validation_history=validation_history,
        best_validation=best_validation,
        best_validation_episode=best_validation_episode,
        best_model_path=best_model_path,
        validation_history_path=validation_history_path if validation_history else None,
    )
    print(f"[train] saved training log -> {log_path}")

    _plot_training_curves(history, curve_path)
    return agent, summary


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train DQN agent on privilege escalation MDP.")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--model-path", type=str, default="artifacts/dqn_model.pt")
    parser.add_argument("--history-path", type=str, default="artifacts/dqn_training_history.csv")
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--validation-episodes", type=int, default=200)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--best-model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    _, summary = train(
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        model_path=args.model_path,
        history_path=args.history_path,
        log_interval=args.log_interval,
        validation_episodes=args.validation_episodes,
        validation_interval=args.validation_interval,
        validation_seed=args.validation_seed,
        best_model_path=args.best_model_path,
        device=args.device,
    )
    print("[train] complete")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    _cli()
