# RL-Guided Post-Access Linux Privilege Escalation Decision Engine

Research prototype for modeling Linux post-access privilege escalation as a finite episodic MDP and comparing:

- Deterministic expert baseline policy
- Deep Q-Network (DQN) policy

This project is a simulator for decision-making research. It does **not** execute real exploits.

## Project Structure

- `actions.py`: action space and vector mappings
- `state.py`: 17-feature observable state representation
- `host.py`: hidden host ground-truth sampling and escalation success model
- `environment.py`: core MDP dynamics, rewards, termination
- `baseline_policy.py`: deterministic expert heuristic
- `evaluation.py`: evaluation and bootstrap comparison utilities
- `dqn_network.py`: DQN model (`17 -> 64 -> 64 -> 11`)
- `replay_buffer.py`: replay memory
- `dqn_agent.py`: DQN agent logic
- `train_dqn.py`: training entrypoint
- `evaluate_dqn.py`: baseline vs DQN evaluation entrypoint
- `main.py`: train + evaluate orchestration

## Run

Train DQN:

```bash
python train_dqn.py --episodes 10000 --seed 42
```

Evaluate trained model:

```bash
python evaluate_dqn.py --model-path artifacts/dqn_model.pt --seeds 7,42,1337 --episodes-per-seed 1000
```

One-shot pipeline:

```bash
python main.py --train-episodes 10000 --episodes-per-seed 1000 --seeds 7,42,1337
```

## Current Experiment Artifacts

- `artifacts/dqn_model.pt`
- `artifacts/dqn_training_history.csv`
- `artifacts/final_report.json`

