# PostEx Agent - RL-Guided Linux Privilege Escalation Decision Engine

Modular post-exploitation decision support system for penetration testers.
It models Linux privilege escalation as an MDP and uses a DQN policy to recommend next actions.

## Architecture

```
Shell Session (manual/metasploit/ssh)
        |
        v
Command Executor   <- safety blocklist enforcement
        |
        v
Parser Engine      <- action-specific structured output parsers
        |
        v
State Builder      <- updates HostState from parsed outputs
        |
        v
RL Decision Engine <- DQN policy: state_vector -> action
        |
        v
Action Selection   <- operator confirm or auto mode
```

## Project Structure

```
postex_agent/
  core/
    state.py
    actions.py
  environment/
    simulation_env.py
    host_archetypes.py
    real_env.py
    command_library.py
    parser_registry.py
    state_builder.py
  parsers/
    base_parser.py
    identity_parser.py
    sudo_parser.py
    suid_parser.py
    capability_parser.py
    cron_parser.py
    writable_parser.py
    kernel_parser.py
    credential_parser.py
  sessions/
    base_session.py
    manual_shell.py
    metasploit_session.py
  rl/
    dqn_network.py
    replay_buffer.py
    dqn_agent.py
    train_dqn.py
    evaluate_dqn.py
    policy_inference.py
    baseline_policy.py
  execution/
    command_executor.py
  cli/
    agent_cli.py
tests/
  test_integration.py
```

## State Representation (35-dim vector)

- `0`: current_privilege
- `1`: os_identified
- `2`: user_identified
- `3..9`: checked flags
- `10..16`: found flags
- `17..23`: exploit failure counts
- `24..30`: vector richness signals
- `31`: credential count
- `32`: credential quality
- `33`: normalized timestep
- `34`: cumulative risk exposure

Vector key order:
`sudo, suid, capabilities, writable_path, cron, credentials, kernel`

## Action Space (16 actions)

- `IDENTIFY_OS`
- `IDENTIFY_USER`
- `CHECK_SUDO`
- `CHECK_SUID`
- `CHECK_CAPABILITIES`
- `CHECK_WRITABLE`
- `CHECK_CRON`
- `SEARCH_CREDENTIALS`
- `CHECK_KERNEL`
- `EXPLOIT_SUDO`
- `EXPLOIT_SUID`
- `EXPLOIT_CAP`
- `EXPLOIT_CRON`
- `EXPLOIT_KERNEL`
- `VERIFY_ROOT`
- `STOP`

## Installation

```bash
pip install -r requirements.txt
```

Optional adapters:

- `paramiko` for SSH session mode
- `pymetasploit3` for Metasploit RPC mode

## Usage

Train:

```bash
python -m postex_agent.rl.train_dqn \
  --episodes 10000 \
  --seed 42 \
  --model-path artifacts/dqn_model.pt \
  --history-path artifacts/dqn_training_history.csv \
  --log-interval 500
```

Evaluate:

```bash
python -m postex_agent.rl.evaluate_dqn \
  --model-path artifacts/dqn_model.pt \
  --seeds 7,42,1337 \
  --episodes-per-seed 1000 \
  --report-path artifacts/eval_report.json
```

## Track 3 Simulator

Track 3 upgrades the training distribution to better reflect real post-exploitation:

- Correlated host archetypes: `neglected_server`, `corporate_app`, `containerized_app`, `hardened_host`, `ctf_lab`
- Vector-specific exploit success rates:
  - `sudo` and `suid` are near-deterministic when found
  - `capabilities` are high confidence and improve with richer findings
  - `cron` becomes significantly stronger when `writable_path` is also found
  - `kernel` remains the genuinely stochastic vector
- Limited retry budget:
  - deterministic vectors are one-shot
  - `kernel` is retryable up to 3 times
- Per-archetype evaluation summaries and Track 3 policy-discipline metrics are written by `evaluate_dqn.py`

Track 3 intentionally keeps the live CLI/session execution path stable. The new realism is in the simulator and evaluation stack so models can be retrained without changing the operator workflow.

Interactive decision support:

```bash
python -m postex_agent.cli.agent_cli --session manual --model-path artifacts/dqn_model.pt
```

SSH mode:

```bash
python -m postex_agent.cli.agent_cli \
  --session ssh \
  --host 10.10.10.5 \
  --user www-data \
  --key ~/.ssh/id_rsa \
  --model-path artifacts/dqn_model.pt
```

Metasploit mode:

```bash
python -m postex_agent.cli.agent_cli \
  --session metasploit \
  --msf-session 1 \
  --msf-host 127.0.0.1 \
  --msf-port 55553 \
  --model-path artifacts/dqn_model.pt
```

## Safety Constraints

The command library enforces blocklist checks and blocks destructive patterns such as:

- `rm -rf`
- `mkfs`
- `shutdown`, `reboot`, `halt`, `poweroff`
- `dd if=`

Session output and decisions are logged as JSONL in `logs/`.

## Tests

```bash
python tests/test_integration.py
python tests/test_e2e_cli.py
```

Expected:

- `All tests passed! [OK]` for the integration suite
- `All E2E checks passed!` for the CLI end-to-end suite

