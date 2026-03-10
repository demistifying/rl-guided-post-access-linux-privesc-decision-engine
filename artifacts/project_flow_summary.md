# PostEx Agent: RL-Guided Linux Privilege Escalation Decision Engine
**Project Flow and Architecture Pipeline Summary**

This document sequentially outlines the entire project flow, the architectural decisions, the reasoning behind each step, and the evolution of the RL model up to the current state.

---

## 1. Project Goal & Scope
The goal of this project is to build a **production-grade, deployable decision-support tool for Linux post-exploitation**. It focuses strictly on the privilege escalation phase on Linux targets.
- **The Problem:** Enumerating a Linux host yields a massive amount of data. Pentesters must decide what to check, in what order, and what to exploit.
- **The Solution:** An RL agent (co-pilot) that runs enumeration commands, parses outputs, builds a structured picture of the host's attack surface, and uses a trained Deep Q-Network (DQN) policy to recommend the optimal next escalation vector.

---

## 2. Core Abstractions & Environment Design
Before any RL code was written, we defined a clean, modular Markov Decision Process (MDP) for the privilege escalation domain.

### A. State Representation (`state.py`)
Initially, the state was a bare-bones **17-dimensional vector** tracking 7 common privilege escalation vectors (Sudo, SUID, Capabilities, Writable Paths, Cron, Credentials, Kernel):
- `current_privilege` (user=0, root=1)
- `os_identified`, `user_identified`
- `checked[7]` (has this vector been enumerated?)
- `found[7]` (is this vector exploitable/present?)

### B. Action Space (`actions.py`)
A discrete continuous action space of **16 actions**:
- 2 Identification actions (OS, User)
- 7 Enumeration actions (Check vector)
- 7 Exploitation actions (Exploit vector)

### C. Simulation Environment (`simulation_env.py`)
To train the RL agent without executing real exploits, we built a stochastic simulation of a Linux host.
- **Ground Truth Properties:** Hosts are generated with specific probabilities for each vector being viable (e.g., SUID=35% base probability).
- **Rewards System:**
  - Useful enumeration: +1.0
  - Redundant action: -1.0
  - Discovery: +3.0
  - Successful escalation (Root): +10.0
  - Failed escalation: -3.0
  - Premature/unenumerated exploit: -2.0
  - Step penalty: -0.1 (encourages speed)

### D. Deterministic Baseline Policy (`baseline_policy.py`)
To measure RL effectiveness, we implemented a hand-crafted heuristic baseline resembling a human pentester. It checks OS/User, then enumerates all vectors in a fixed lowest-risk-first order, executing exploits as soon as a vulnerable vector is found.

---

## 3. Infrastructure & Testing Pipeline
To ensure production readiness, we established a rigorous testing and evaluation pipeline.
- **Integration Tests:** 28 tests (`tests/test_integration.py`) covering state transitions, rewards, and baseline behaviors.
- **Evaluation Loop:** A rigorous multi-seed evaluation script (`evaluate_dqn.py`) running 1000 greedy episodes per seed. It collects bootstrap 95% Confidence Intervals comparing the RL agent against the baseline across success rate, average steps, mean reward, and risk exposure.

---

## 4. Initial RL Implementation (Vanilla DQN)
We implemented a standard Deep Q-Network (`dqn_agent.py`, `dqn_network.py`) to learn the optimal policy.
- **Result:** The vanilla DQN underperformed. It achieved a **78.5% success rate** vs the baseline's **82.2%**.
- **Root Causes Discovered:**
  1. *Redundant Actions:* The agent executed already-checked/exploited actions (averaging 1.19 redundant actions per episode), wasting steps.
  2. *Small Replay Buffer:* The 10k buffer only held ~1000 episodes, "forgetting" old successful learning.
  3. *No Prioritization:* Sparse rewards (escalating to root) weren't sampled often enough.

---

## 5. Iterative DQN Improvements
Based on the root causes, we embarked on a series of architectural upgrades.

### Iteration 1: Action Masking, Larger Buffer, & PER (v2)
- **Action Masking:** Added `action_mask()` to prevent the DQN from selecting invalid or mathematically useless actions by overwriting their Q-values with `-inf`. This dropped redundant actions from 1.19 to **0.00**.
- **Buffer Upgrade:** Increased buffer size from 10k to 50k transitions.
- **PER (Prioritized Experience Replay):** Implemented a SumTree-based buffer (`replay_buffer.py`) to oversample trajectories with high Temporal Difference (TD) error (surprising/valuable transitions), applying Importance Sampling weights to correct bias.
- **Result:** Success rate jumped to **83.0%** (beating the baseline).

### Iteration 2: Double DQN (v3)
- **Change:** Standard DQN severely overestimates Q-values. We decoupled action selection from evaluation. The *online* network selects the best action, and the *target* network evaluates its Q-value.
- **Result:** Stabilized training trajectory and maintained the **83.0%** success rate.

---

## 6. Overcoming Environment Limitations
**The Problem:** The 17-dimensional state was simply too sparse. It didn't track *how many* exploits failed, *how rich* the attack vectors were, or whether credentials found were mere hashes vs root passwords.

### Iteration 3: Expanding State Vector to 35 Dimensions (v4)
We expanded the feature space to give the DQN actionable intelligence that hand-coded heuristics struggle to parse:
- `exploit_failures[7]` (Prevents infinite exploit loops on failed targets)
- `richness[7]` (Number of items found per vector, modulating success likelihood)
- `cred_count` & `cred_quality` (Is it an SSH key vs a bcrypt hash?)
- `time_step` (Awareness of remaining episode duration)
- `cumulative_risk` (Awareness of risk budget)

**Simulation Update:** The simulation was made harder to account for these variables. Exploit success probabilities are now explicitly tied to vector `richness` and `cred_quality`.
- **Result:** The baseline heuristic's success dropped slightly (82.6%) because the environment got harder, yet the newly informed 35-dim DQN achieved **83.1%**, proving it could leverage the complex probabilistic features better than a rigid script.

---

## 7. The Final Architecture: Dueling DQN (v5)
- **Change:** We replaced the sequential multi-layer perceptron with a **Dueling Network Architecture**. `Q(s,a)` is now split into two streams:
  1. **Value Stream `V(s)`:** "How inherently good is this state?" (e.g., finding root credentials is good regardless of the next action).
  2. **Advantage Stream `A(s,a)`:** "Given this state, how much better is action X than the others?"
  These are recombined: `Q(s,a) = V(s) + A(s,a) - mean(A)`.
- **Reasoning:** In privesc, early enumeration states all roughly hold equal value regardless of the action chosen. Dueling allows the network to learn state-values much faster without updating 16 independent action Q-values.

**Final Result (Current State):**
The Dueling DQN achieves the project's highest win-rate (**83.2%** vs the baseline's **82.6%**) in a highly complex stochastic simulation environment without selecting a single redundant action.

### Current Pipeline Overview
1. **Simulation Env (`simulation_env.py`)** -> generates 35-dim `HostState`.
2. **Action Mask (`actions.py`)** -> forces valid actions.
3. **Agent (`dqn_agent.py`)** -> Double DQN with ε-greedy + PER (50k capacity).
4. **Network (`dqn_network.py`)** -> Dueling DQN structure.
5. **Trainer (`train_dqn.py`)** -> 10,000 episodes training loop.
6. **Inference / Deployment (`policy_inference.py`)** -> Greedy action selection via `agent.select_action(state, explore=False, mask=mask)`.
