"""
PostEx Agent CLI - RL-guided Linux privilege escalation decision support.

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

from postex_agent.core.actions import ACTION_DESCRIPTIONS, Action
from postex_agent.core.state import HostState, VECTOR_KEYS
from postex_agent.execution.live_runtime import LiveExecutionController
from postex_agent.rl.policy_inference import RLPolicy

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
GREY = "\033[90m"


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


def _build_session(args: argparse.Namespace):
    kind = args.session.lower()

    if kind == "manual":
        from postex_agent.sessions.manual_shell import ManualShellSession

        print(_c("[*] Starting local shell session...", CYAN))
        return ManualShellSession(shell=args.shell or None)

    if kind == "metasploit":
        from postex_agent.sessions.metasploit_session import MetasploitSession

        print(
            _c(
                f"[*] Connecting to msfrpcd at {args.msf_host}:{args.msf_port} "
                f"(session {args.msf_session})...",
                CYAN,
            )
        )
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


def _print_banner() -> None:
    print(
        _c(
            """
+---------------------------------------------------------------+
| RL-Guided Post-Exploitation Decision Engine                   |
| Linux Privilege Escalation Assistant                          |
+---------------------------------------------------------------+
""",
            CYAN,
        )
    )


def _print_state(state: HostState) -> None:
    privilege = _c("ROOT", GREEN) if state.current_privilege == 1 else _c("user", YELLOW)
    user = state.current_user or "?"
    os_name = state.os_info or "?"
    kernel = state.kernel_version or "?"

    print(f"\n{'-' * 72}")
    print(
        f"  {_c('Privilege:', BOLD)} {privilege}  |  "
        f"{_c('User:', BOLD)} {user}  |  "
        f"{_c('OS:', BOLD)} {os_name}  |  "
        f"{_c('Kernel:', BOLD)} {kernel}"
    )
    checked = [f"{v[:6]}:{_c('Y', GREEN) if state.checked[v] else _c('.', GREY)}" for v in VECTOR_KEYS]
    found = [f"{v[:6]}:{_c('Y', GREEN) if state.found[v] else _c('.', GREY)}" for v in VECTOR_KEYS]
    print(f"  {_c('Checked:', BOLD)} " + "  ".join(checked))
    print(f"  {_c('Found:  ', BOLD)} " + "  ".join(found))
    print(f"{'-' * 72}")


def _print_suggestion(
    action: Action,
    commands: List[str],
    top_actions: Optional[list] = None,
) -> None:
    desc = ACTION_DESCRIPTIONS.get(action, "")
    colour = YELLOW if action == Action.STOP else GREEN

    print(f"\n  {_c('> RL Suggested Action:', BOLD)} {_c(action.name, colour)}")
    print(f"  {_c('Description:', GREY)} {desc}")

    if commands:
        print(f"  {_c('Commands:', GREY)}")
        for cmd in commands:
            line_colour = GREY if cmd.strip().startswith("#") else CYAN
            print(f"    {_c('$', CYAN)} {_c(cmd, line_colour)}")
    else:
        print(f"  {_c('  (no commands - decision-only action)', GREY)}")

    if top_actions and len(top_actions) > 1:
        print(f"  {_c('Alternatives:', GREY)}")
        for alt, qv, alt_desc in top_actions[1:3]:
            print(f"    [{qv:+.2f}] {alt.name} - {alt_desc[:60]}")


def _print_parsed(parsed: dict) -> None:
    found = parsed.get("vector_found", False)
    details = parsed.get("details", {})
    label = _c("FOUND", GREEN) if found else _c("not found", GREY)
    print(f"\n  {_c('Parser result:', BOLD)} {label}")
    for key, value in details.items():
        if not value or key == "raw":
            continue
        if isinstance(value, list):
            preview = ", ".join(str(item)[:80] for item in value[:5])
            print(f"    {_c(key + ':', GREY)} {preview}")
        else:
            print(f"    {_c(key + ':', GREY)} {str(value)[:120]}")


def _prompt(msg: str, default: str = "y") -> str:
    try:
        answer = input(msg).strip()
        return answer if answer else default
    except (EOFError, KeyboardInterrupt):
        return "n"


def _manual_select() -> Action:
    print("\n  Available actions:")
    for action in Action:
        print(f"    [{action.value:>2d}] {action.name}")

    while True:
        raw = _prompt("  Select action ID: ", str(Action.STOP.value))
        try:
            return Action(int(raw))
        except (ValueError, KeyError):
            print("  Invalid. Try again.")


MAX_STEPS = 30


def run_agent(
    policy: RLPolicy,
    session,
    auto: bool = False,
    max_steps: int = MAX_STEPS,
    log_dir: str = "logs",
    report_out: Optional[str] = None,
) -> None:
    logger = StepLogger(log_dir=log_dir)
    runtime = LiveExecutionController(
        session=session,
        max_steps=max_steps,
        log_path=os.path.join(log_dir, "execution.jsonl"),
    )

    print(_c(f"\n[*] Session log: {logger.path}", GREY))
    print(_c("[*] Starting agent loop. Press Ctrl+C to abort.\n", GREY))

    for step in range(1, max_steps + 1):
        state = runtime.current_state

        print(f"\n{_c(f'Step {step}/{max_steps}', BOLD)}")
        _print_state(state)

        if state.current_privilege == 1:
            print(_c("\n[OK] Root access achieved. Stopping.", GREEN))
            break

        state_vec = state.to_vector()
        suggested_action = Action(int(policy.predict(state_vec)))
        top_actions = policy.top_actions(state_vec, n=3)
        commands = runtime.preview_commands(suggested_action)
        _print_suggestion(suggested_action, commands, top_actions)

        selected_action = suggested_action

        if suggested_action == Action.STOP:
            print(_c("\n[*] Policy suggests STOP.", YELLOW))
            if auto or _prompt("  Stop? [Y/n]: ", "y").lower() in ("y", ""):
                break
            selected_action = _manual_select()
            commands = runtime.preview_commands(selected_action)

        if not auto:
            answer = _prompt("\n  Execute? [Y/n/s(skip)/q(quit)]: ", "y").lower()
            if answer == "q":
                print(_c("[*] Aborted by operator.", YELLOW))
                break
            if answer == "s":
                print(_c("[*] Skipping.", GREY))
                continue
            if answer not in ("y", ""):
                selected_action = _manual_select()
                commands = runtime.preview_commands(selected_action)

        _, _, _, info = runtime.step(selected_action)
        state = runtime.current_state

        for result in info["execution"]:
            print(f"\n  {_c('$', CYAN)} {result['command']}")
            if result["blocked"]:
                print(_c(f"  [BLOCKED] {result['error']}", RED))
                continue
            if result["error"]:
                print(_c(f"  [STDERR] {result['error'][:200]}", YELLOW))

            output = result.get("output", "")
            lines = output.splitlines()
            for line in lines[:30]:
                print(f"  {_c('|', GREY)} {line}")
            if len(lines) > 30:
                print(_c(f"  | ... ({len(lines) - 30} more lines)", GREY))

        _print_parsed(info["parsed"])
        logger.log(
            {
                "step": step,
                "state_before": info["state_before"],
                "action": info["action"],
                "commands": info["commands"],
                "parsed": info["parsed"],
                "state_after": info["state_after"],
            }
        )

        if selected_action == Action.VERIFY_ROOT and state.current_privilege == 1:
            print(_c("\n[OK] Root confirmed.", GREEN))
            break

    print(_c(f"\n[*] Agent loop complete. Log: {logger.path}", GREY))
    _print_state(runtime.current_state)

    if report_out:
        from postex_agent.cli.reporter import EngagementReporter
        reporter = EngagementReporter(logger.path)
        report_md = reporter.generate()
        with open(report_out, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(_c(f"[*] Saved Engagement Report to: {report_out}", GREEN))


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="postex-agent",
        description="RL-guided Linux post-exploitation decision support agent.",
    )
    parser.add_argument("--session", choices=["manual", "metasploit", "ssh"], default="manual")
    parser.add_argument("--model-path", default="artifacts/dqn_model.pt")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Non-interactive: execute all RL suggestions automatically.",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--report-out", default=None, help="Save markdown engagement report to this path")
    parser.add_argument("--device", default=None)

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default=None)
    parser.add_argument("--key", default=None, help="Path to SSH private key")
    parser.add_argument("--port", type=int, default=22)

    parser.add_argument("--msf-session", type=int, default=1)
    parser.add_argument("--msf-host", default="127.0.0.1")
    parser.add_argument("--msf-port", type=int, default=55553)
    parser.add_argument("--msf-password", default="msf")

    parser.add_argument(
        "--shell",
        default=None,
        help="Shell binary for manual session (default: /bin/sh)",
    )
    return parser


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
            report_out=args.report_out,
        )
    except KeyboardInterrupt:
        print(_c("\n[*] Interrupted.", YELLOW))
    finally:
        session.close()


if __name__ == "__main__":
    main()
