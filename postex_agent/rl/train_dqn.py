"""
DQN training entrypoint.
Trains the agent on the SimulationEnv and saves:
  - model checkpoint  (artifacts/dqn_model.pt)
  - per-episode CSV   (artifacts/dqn_training_history.csv)
  - training log      (artifacts/training_log.txt)
  - training curves   (artifacts/training_curve.png)
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


# ── Utilities ────────────────────────────────────────────────────────────

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


# ── Training log writer ─────────────────────────────────────────────────

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
) -> None:
    """Write a detailed human-readable training log to *log_path*."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Compute milestone rolling metrics
    milestones = []
    windows = [100, 500, 1000]
    for w in windows:
        if len(history) >= w:
            milestones.append(
                f"  Last {w:>5d} eps:  "
                f"success={_rolling_mean(history, 'success', w):.4f}  "
                f"reward={_rolling_mean(history, 'reward', w):>8.3f}  "
                f"steps={_rolling_mean(history, 'steps', w):>6.2f}  "
                f"loss={_rolling_mean(history, 'mean_loss', w):>7.4f}  "
                f"risk={_rolling_mean(history, 'risk_exposure', w):>6.4f}"
            )

    # Best / worst episode
    best_ep  = max(history, key=lambda r: r["reward"])
    worst_ep = min(history, key=lambda r: r["reward"])

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("  PostEx Agent — DQN Training Log\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"  Timestamp      : {ts}\n")
        f.write(f"  Duration       : {elapsed_sec:.1f}s ({elapsed_sec/60:.1f}m)\n")
        f.write(f"  Device         : {device}\n\n")

        # ── Hyperparameters
        f.write("-" * 72 + "\n")
        f.write("  HYPERPARAMETERS\n")
        f.write("-" * 72 + "\n")
        f.write(f"  Episodes              : {episodes}\n")
        f.write(f"  Seed                  : {seed}\n")
        f.write(f"  Max steps / episode   : {max_steps}\n")
        f.write(f"  Network               : {config.state_dim}→{config.hidden_dim}→{config.hidden_dim}→{config.action_dim}\n")
        f.write(f"  Gamma                 : {config.gamma}\n")
        f.write(f"  Learning rate         : {config.learning_rate}\n")
        f.write(f"  Replay buffer size    : {config.replay_buffer_size}\n")
        f.write(f"  Batch size            : {config.batch_size}\n")
        f.write(f"  Target net update     : every {config.target_update_interval} steps\n")
        f.write(f"  Epsilon schedule      : {config.epsilon_start}→{config.epsilon_end} over {config.epsilon_decay_episodes} eps\n\n")

        # ── Per-interval progress
        f.write("-" * 72 + "\n")
        f.write("  TRAINING PROGRESS\n")
        f.write("-" * 72 + "\n")
        f.write(f"  {'Episode':>8s}  {'Epsilon':>8s}  {'Success':>8s}  {'Reward':>8s}  {'Steps':>8s}  {'Loss':>10s}  {'Risk':>8s}  {'Redundant':>10s}\n")
        f.write(f"  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'----------':>10s}  {'--------':>8s}  {'----------':>10s}\n")
        for line in interval_logs:
            f.write(f"  {line}\n")
        f.write("\n")

        # ── Rolling-window summary
        f.write("-" * 72 + "\n")
        f.write("  ROLLING-WINDOW SUMMARY\n")
        f.write("-" * 72 + "\n")
        for m in milestones:
            f.write(m + "\n")
        f.write("\n")

        # ── Overall summary
        f.write("-" * 72 + "\n")
        f.write("  FINAL RESULTS\n")
        f.write("-" * 72 + "\n")
        for k, v in summary.items():
            f.write(f"  {k:<20s}: {v:.4f}\n" if isinstance(v, float) else f"  {k:<20s}: {v}\n")
        f.write("\n")

        # ── Best / worst
        f.write("-" * 72 + "\n")
        f.write("  NOTABLE EPISODES\n")
        f.write("-" * 72 + "\n")
        f.write(f"  Best  reward : ep {int(best_ep['episode']):>6d}  reward={best_ep['reward']:>8.2f}  steps={int(best_ep['steps']):>3d}  success={int(best_ep['success'])}\n")
        f.write(f"  Worst reward : ep {int(worst_ep['episode']):>6d}  reward={worst_ep['reward']:>8.2f}  steps={int(worst_ep['steps']):>3d}  success={int(worst_ep['success'])}\n\n")

        # ── Output files
        f.write("-" * 72 + "\n")
        f.write("  OUTPUT FILES\n")
        f.write("-" * 72 + "\n")
        f.write(f"  Model checkpoint : {model_path}\n")
        f.write(f"  Training CSV     : {history_path}\n")
        f.write(f"  Training log     : {log_path}\n")
        curve_path = os.path.join(os.path.dirname(log_path), "training_curve.png")
        f.write(f"  Training curve   : {curve_path}\n\n")
        f.write("=" * 72 + "\n")


# ── Training curve plotter ───────────────────────────────────────────────

def _plot_training_curves(history: List[Dict[str, float]], output_path: str) -> None:
    """Generate a 4-panel training curve PNG from episode history."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[train] matplotlib not available — skipping curve plot")
        return

    episodes = [int(h["episode"]) for h in history]
    window = min(500, len(history) // 4) or 1

    def _rolling(key: str) -> List[float]:
        vals = [h[key] for h in history]
        out = []
        for i in range(len(vals)):
            start = max(0, i - window + 1)
            out.append(float(np.mean(vals[start : i + 1])))
        return out

    r_success = _rolling("success")
    r_reward  = _rolling("reward")
    r_steps   = _rolling("steps")
    r_loss    = _rolling("mean_loss")
    epsilons  = [h["epsilon"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"DQN Training Curves ({len(history):,} Episodes)",
        fontsize=14, fontweight="bold",
    )

    # ── Success rate
    ax = axes[0, 0]
    ax.plot(episodes, r_success, color="#2ecc71", linewidth=1.5)
    ax.axhline(y=0.85, color="#e74c3c", linestyle="--", alpha=0.7, label="Baseline (85%)")
    ax.set_title(f"Success Rate ({window}-ep rolling)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Mean reward
    ax = axes[0, 1]
    ax.plot(episodes, r_reward, color="#3498db", linewidth=1.5)
    ax.set_title(f"Mean Reward ({window}-ep rolling)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    # ── Mean steps
    ax = axes[1, 0]
    ax.plot(episodes, r_steps, color="#e67e22", linewidth=1.5)
    ax.set_title(f"Mean Steps per Episode ({window}-ep rolling)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.grid(True, alpha=0.3)

    # ── Loss + epsilon
    ax = axes[1, 1]
    ax.plot(episodes, r_loss, color="#9b59b6", linewidth=1.5, label="Loss")
    ax2 = ax.twinx()
    ax2.plot(episodes, epsilons, color="#95a5a6", linewidth=1, alpha=0.6, label="Epsilon")
    ax2.set_ylabel("Epsilon", color="#95a5a6")
    ax.set_title("Mean Loss & Epsilon Schedule")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] saved training curve → {output_path}")


# ── Main training function ───────────────────────────────────────────────

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
    resolved_device = str(agent.device)

    total_steps = 0
    history: List[Dict[str, float]] = []
    interval_logs: List[str] = []
    t_start = time.time()

    for ep in range(episodes):
        state = env.reset()
        mask  = env.action_mask()
        agent.set_epsilon(ep)
        agent.set_beta(ep)
        done   = False
        losses: List[float] = []

        while not done:
            action     = agent.select_action(state, explore=True, mask=mask)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            state        = next_state
            mask         = info.get("action_mask", mask)
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
            succ = _rolling_mean(history, "success", w)
            rew  = _rolling_mean(history, "reward", w)
            stp  = _rolling_mean(history, "steps", w)
            lss  = _rolling_mean(history, "mean_loss", w)
            rsk  = _rolling_mean(history, "risk_exposure", w)
            red  = _rolling_mean(history, "redundant_actions", w)

            log_line = (
                f"[train] ep={ep+1:>6d} "
                f"eps={agent.epsilon:.3f} "
                f"success={succ:.3f} "
                f"steps={stp:.2f} "
                f"reward={rew:.3f}"
            )
            print(log_line)

            # Structured row for the log file
            interval_logs.append(
                f"{ep+1:>8d}  {agent.epsilon:>8.3f}  {succ:>8.4f}  {rew:>8.3f}  {stp:>8.2f}  {lss:>10.4f}  {rsk:>8.4f}  {red:>10.2f}"
            )

    elapsed = time.time() - t_start

    # ── Save core artifacts
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

    # ── Write detailed training log
    artifacts_dir = os.path.dirname(model_path) or "artifacts"
    log_path   = os.path.join(artifacts_dir, "training_log.txt")
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
    )
    print(f"[train] saved training log  → {log_path}")

    # ── Plot training curves
    _plot_training_curves(history, curve_path)

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

