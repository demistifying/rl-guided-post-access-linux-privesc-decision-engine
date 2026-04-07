"""
PostEx Agent CLI — RL-guided Linux privilege escalation decision support.

Usage:
    python -m postex_agent.cli.agent_cli --session manual
    python -m postex_agent.cli.agent_cli --session ssh --host 10.10.10.5 --user www-data
    python -m postex_agent.cli.agent_cli --session metasploit --msf-session 1
    python -m postex_agent.cli.agent_cli --session manual --auto
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import List, Optional

from postex_agent.core.actions import (
    ACTION_DESCRIPTIONS, Action, VECTOR_BY_EXPLOIT_ACTION,
)
from postex_agent.core.state import HostState, VECTOR_KEYS
from postex_agent.environment.command_library import get_commands
from postex_agent.environment.parser_registry import parse_output
from postex_agent.environment.state_builder import (
    update_state, update_temporal, VECTOR_RISK_PENALTIES,
)
from postex_agent.execution.command_executor import CommandExecutor
from postex_agent.rl.policy_inference import RLPolicy

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
GREY   = "\033[90m"


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


# ── Session factory ───────────────────────────────────────────────────────────

def _build_session(args: argparse.Namespace):
    kind = args.session.lower()

    if kind == "manual":
        from postex_agent.sessions.manual_shell import ManualShellSession
        print(_c("[*] Starting local shell session...", CYAN))
        return ManualShellSession(shell=args.shell or None)

    if kind == "metasploit":
        from postex_agent.sessions.metasploit_session import MetasploitSession
        print(_c(
            f"[*] Connecting to msfrpcd at {args.msf_host}:{args.msf_port} "
            f"(session {args.msf_session})...", CYAN
        ))
        return MetasploitSession(
            session_id=args.msf_session,
            host=args.msf_host,
            port=args.msf_port,
            password=args.msf_password,
        )

    if kind == "ssh":
        from postex_agent.sessions.metasploit_session import SSHSession
        print(_c(f"[*] Connecting via SSH to {args.host}:{args.port}...", CYAN))
        return SSHSession(
            host=args.host,
            username=args.user,
            password=args.password,
            key_path=args.key,
            port=args.port,
        )

    raise ValueError(f"Unknown session type: {kind}")


# ── Step logger ───────────────────────────────────────────────────────────────

class StepLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(log_dir, f"session_{ts}.jsonl")

    def log(self, entry: dict) -> None:
        entry["ts"] = datetime.datetime.utcnow().isoformat() + "Z"
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    @property
    def path(self) -> str:
        return self._path


# ── Display helpers ───────────────────────────────────────────────────────────

def _print_banner() -> None:
    print(_c("""
+---------------------------------------------------------------+
| RL-Guided Post-Exploitation Decision Engine                   |
| Linux Privilege Escalation Assistant                          |
+---------------------------------------------------------------+
""", CYAN))


def _print_state(state: HostState) -> None:
    priv   = _c("ROOT", GREEN) if state.current_privilege == 1 else _c("user", YELLOW)
    user   = state.current_user or "?"
    os_    = state.os_info or "?"
    kernel = state.kernel_version or "?"

    print(f"\n{'-' * 72}")
    print(
        f"  {_c('Privilege:', BOLD)} {priv}  |  "
        f"{_c('User:', BOLD)} {user}  |  "
        f"{_c('OS:', BOLD)} {os_}  |  "
        f"{_c('Kernel:', BOLD)} {kernel}"
    )
    chk = [f"{v[:6]}:{_c('Y', GREEN) if state.checked[v] else _c('.', GREY)}" for v in VECTOR_KEYS]
    fnd = [f"{v[:6]}:{_c('Y', GREEN) if state.found[v]   else _c('.', GREY)}" for v in VECTOR_KEYS]
    print(f"  {_c('Checked:', BOLD)} " + "  ".join(chk))
    print(f"  {_c('Found:  ', BOLD)} " + "  ".join(fnd))
    print(f"{'-' * 72}")


def _print_suggestion(
    action:      Action,
    commands:    List[str],
    top_actions: Optional[list] = None,
) -> None:
    desc  = ACTION_DESCRIPTIONS.get(action, "")
    color = YELLOW if action == Action.STOP else GREEN

    print(f"\n  {_c('> RL Suggested Action:', BOLD)} {_c(action.name, color)}")
    print(f"  {_c('Description:', GREY)} {desc}")

    if commands:
        print(f"  {_c('Commands:', GREY)}")
        for cmd in commands:
            line_color = GREY if cmd.strip().startswith("#") else CYAN
            print(f"    {_c('$', CYAN)} {_c(cmd, line_color)}")
    else:
        print(f"  {_c('  (no commands — decision-only action)', GREY)}")

    if top_actions and len(top_actions) > 1:
        print(f"  {_c('Alternatives:', GREY)}")
        for alt, qv, alt_desc in top_actions[1:3]:
            print(f"    [{qv:+.2f}] {alt.name} — {alt_desc[:60]}")


def _print_parsed(parsed: dict) -> None:
    found   = parsed.get("vector_found", False)
    details = parsed.get("details", {})
    label   = _c("FOUND", GREEN) if found else _c("not found", GREY)
    print(f"\n  {_c('Parser result:', BOLD)} {label}")
    for key, val in details.items():
        if not val or key == "raw":
            continue
        if isinstance(val, list):
            preview = ", ".join(str(v)[:80] for v in val[:5])
            print(f"    {_c(key + ':', GREY)} {preview}")
        else:
            print(f"    {_c(key + ':', GREY)} {str(val)[:120]}")


def _prompt(msg: str, default: str = "y") -> str:
    try:
        ans = input(msg).strip()
        return ans if ans else default
    except (EOFError, KeyboardInterrupt):
        return "n"


def _manual_select() -> Action:
    print("\n  Available actions:")
    for a in Action:
        print(f"    [{a.value:>2d}] {a.name}")
    while True:
        raw = _prompt("  Select action ID: ", str(Action.STOP.value))
        try:
            return Action(int(raw))
        except (ValueError, KeyError):
            print("  Invalid. Try again.")


# ── Core agent loop ───────────────────────────────────────────────────────────

MAX_STEPS = 30


def run_agent(
    policy:    RLPolicy,
    session,
    auto:      bool = False,
    max_steps: int  = MAX_STEPS,
    log_dir:   str  = "logs",
) -> None:
    logger          = StepLogger(log_dir=log_dir)
    executor        = CommandExecutor(session=session, log_path=os.path.join(log_dir, "execution.jsonl"))
    state           = HostState()
    cumulative_risk = 0.0

    print(_c(f"\n[*] Session log: {logger.path}", GREY))
    print(_c("[*] Starting agent loop. Press Ctrl+C to abort.\n", GREY))

    for step in range(1, max_steps + 1):
        print(f"\n{_c(f'Step {step}/{max_steps}', BOLD)}")
        _print_state(state)

        if state.current_privilege == 1:
            print(_c("\n[OK] Root access achieved. Stopping.", GREEN))
            break

        # Policy inference
        state_vec   = state.to_vector()
        action      = policy.predict(state_vec)
        top_actions = policy.top_actions(state_vec, n=3)

        # Build commands — pass state so exploit actions get contextual one-liners
        commands = get_commands(action, state=state)
        _print_suggestion(action, commands, top_actions)

        # Handle STOP suggestion
        if action == Action.STOP:
            print(_c("\n[*] Policy suggests STOP.", YELLOW))
            if auto or _prompt("  Stop? [Y/n]: ", "y").lower() in ("y", ""):
                break
            action   = _manual_select()
            commands = get_commands(action, state=state)

        # Operator confirmation
        if not auto:
            ans = _prompt("\n  Execute? [Y/n/s(skip)/q(quit)]: ", "y").lower()
            if ans == "q":
                print(_c("[*] Aborted by operator.", YELLOW))
                break
            if ans == "s":
                print(_c("[*] Skipping.", GREY))
                continue
            if ans not in ("y", ""):
                action   = _manual_select()
                commands = get_commands(action, state=state)

        # Execute each command (skip comment lines — they are informational only)
        combined_output = ""
        for cmd in commands:
            if cmd.strip().startswith("#"):
                continue
            print(f"\n  {_c('$', CYAN)} {cmd}")
            result = executor.execute(cmd, action_name=action.name)
            if result["blocked"]:
                print(_c(f"  [BLOCKED] {result['error']}", RED))
                continue
            if result["error"]:
                print(_c(f"  [STDERR] {result['error'][:200]}", YELLOW))
            out = result["output"]
            combined_output += out + "\n"
            lines = out.splitlines()
            for line in lines[:30]:
                print(f"  {_c('|', GREY)} {line}")
            if len(lines) > 30:
                print(_c(f"  | ... ({len(lines) - 30} more lines)", GREY))

        # Parse output and update state
        parsed       = parse_output(action, combined_output)
        state_before = state.summary()
        _print_parsed(parsed)
        update_state(state, action, parsed)

        # Track cumulative risk for exploit actions
        if action in VECTOR_BY_EXPLOIT_ACTION:
            vector = VECTOR_BY_EXPLOIT_ACTION[action]
            cumulative_risk += VECTOR_RISK_PENALTIES.get(vector, 0.0)

        # Update temporal features (time_step, cumulative_risk)
        update_temporal(state, step, max_steps, cumulative_risk)

        logger.log({
            "step":         step,
            "state_before": state_before,
            "action":       action.name,
            "commands":     commands,
            "parsed":       parsed,
            "state_after":  state.summary(),
        })

        if action == Action.VERIFY_ROOT and state.current_privilege == 1:
            print(_c("\n[OK] Root confirmed.", GREEN))
            break

    print(_c(f"\n[*] Agent loop complete. Log: {logger.path}", GREY))
    _print_state(state)


# ── CLI argument parser ───────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="postex-agent",
        description="RL-guided Linux post-exploitation decision support agent.",
    )
    p.add_argument("--session",      choices=["manual", "metasploit", "ssh"], default="manual")
    p.add_argument("--model-path",   default="artifacts/dqn_model.pt")
    p.add_argument("--auto",         action="store_true",
                   help="Non-interactive: execute all RL suggestions automatically.")
    p.add_argument("--max-steps",    type=int, default=MAX_STEPS)
    p.add_argument("--log-dir",      default="logs")
    p.add_argument("--device",       default=None)

    # SSH options
    p.add_argument("--host",     default="127.0.0.1")
    p.add_argument("--user",     default="root")
    p.add_argument("--password", default=None)
    p.add_argument("--key",      default=None, help="Path to SSH private key")
    p.add_argument("--port",     type=int, default=22)

    # Metasploit options
    p.add_argument("--msf-session",  type=int, default=1)
    p.add_argument("--msf-host",     default="127.0.0.1")
    p.add_argument("--msf-port",     type=int, default=55553)
    p.add_argument("--msf-password", default="msf")

    # Local shell override
    p.add_argument("--shell", default=None, help="Shell binary for manual session (default: /bin/sh)")
    return p


def main() -> None:
    _print_banner()
    args = _build_argparser().parse_args()

    print(_c(f"[*] Loading RL policy from: {args.model_path}", CYAN))
    try:
        policy = RLPolicy(model_path=args.model_path, device=args.device)
        print(_c("[OK] Policy loaded.", GREEN))
    except FileNotFoundError as exc:
        print(_c(f"[!] {exc}", RED))
        print(_c("[*] Train first: python -m postex_agent.rl.train_dqn --episodes 10000", YELLOW))
        sys.exit(1)
    except Exception as exc:
        print(_c(f"[!] Failed to load policy: {exc}", RED))
        print(_c("[*] This usually means the checkpoint is from an older model shape.", YELLOW))
        print(_c("[*] Retrain or pass a compatible model via --model-path.", YELLOW))
        sys.exit(1)

    try:
        session = _build_session(args)
    except Exception as exc:
        print(_c(f"[!] Session error: {exc}", RED))
        sys.exit(1)

    try:
        run_agent(
            policy=policy,
            session=session,
            auto=args.auto,
            max_steps=args.max_steps,
            log_dir=args.log_dir,
        )
    except KeyboardInterrupt:
        print(_c("\n[*] Interrupted.", YELLOW))
    finally:
        session.close()


if __name__ == "__main__":
    main()
