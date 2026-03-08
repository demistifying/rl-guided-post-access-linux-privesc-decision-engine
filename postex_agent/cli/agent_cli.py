"""
PostEx Agent CLI
Interactive RL-guided privilege escalation decision support tool.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from typing import Optional

from postex_agent.core.actions import ACTION_DESCRIPTIONS, Action
from postex_agent.core.state import HostState, VECTOR_KEYS
from postex_agent.environment.command_library import get_commands
from postex_agent.environment.parser_registry import parse_output
from postex_agent.environment.state_builder import update_state
from postex_agent.execution.command_executor import CommandExecutor
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
    session_type = args.session.lower()

    if session_type == "manual":
        from postex_agent.sessions.manual_shell import ManualShellSession

        shell = args.shell if args.shell else None
        print(_c("[*] Starting local shell session...", CYAN))
        return ManualShellSession(shell=shell)

    if session_type == "metasploit":
        from postex_agent.sessions.metasploit_session import MetasploitSession

        print(
            _c(
                f"[*] Connecting to msfrpcd at {args.msf_host}:{args.msf_port} (session {args.msf_session})...",
                CYAN,
            )
        )
        return MetasploitSession(
            session_id=args.msf_session,
            host=args.msf_host,
            port=args.msf_port,
            password=args.msf_password,
        )

    if session_type == "ssh":
        from postex_agent.sessions.metasploit_session import SSHSession

        print(_c(f"[*] Connecting via SSH to {args.host}:{args.port}...", CYAN))
        return SSHSession(
            host=args.host,
            username=args.user,
            password=args.password,
            key_path=args.key,
            port=args.port,
        )

    raise ValueError(f"Unknown session type: {session_type}")


class StepLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(log_dir, f"session_{ts}.jsonl")

    def log(self, entry: dict) -> None:
        entry["ts"] = datetime.datetime.utcnow().isoformat() + "Z"
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")

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
    os_info = state.os_info or "?"
    kernel = state.kernel_version or "?"

    print(f"\n{'-' * 72}")
    print(
        f"  {_c('Privilege:', BOLD)} {privilege}  |  "
        f"{_c('User:', BOLD)} {user}  |  "
        f"{_c('OS:', BOLD)} {os_info}  |  "
        f"{_c('Kernel:', BOLD)} {kernel}"
    )

    check_row = []
    found_row = []
    for vector in VECTOR_KEYS:
        checked = _c("Y", GREEN) if state.checked[vector] else _c(".", GREY)
        found = _c("Y", GREEN) if state.found[vector] else _c(".", GREY)
        check_row.append(f"{vector[:6]}:{checked}")
        found_row.append(f"{vector[:6]}:{found}")

    print(f"  {_c('Checked:', BOLD)} " + "  ".join(check_row))
    print(f"  {_c('Found:  ', BOLD)} " + "  ".join(found_row))
    print(f"{'-' * 72}")


def _print_action_suggestion(action: Action, q_vals: Optional[list] = None) -> None:
    desc = ACTION_DESCRIPTIONS.get(action, "")
    cmds = get_commands(action)
    action_color = GREEN if action not in (Action.STOP,) else YELLOW

    print(f"\n  {_c('> RL Suggested Action:', BOLD)} {_c(action.name, action_color)}")
    print(f"  {_c('Description:', GREY)} {desc}")

    if cmds:
        print(f"  {_c('Commands:', GREY)}")
        for cmd in cmds:
            print(f"    {_c('$', CYAN)} {cmd}")

    if q_vals:
        print(f"  {_c('Top alternatives:', GREY)}")
        for alt_action, qv, alt_desc in q_vals[1:3]:
            print(f"    [{qv:+.2f}] {alt_action.name} - {alt_desc[:60]}")


def _print_parsed_result(parsed: dict) -> None:
    found = parsed.get("vector_found", False)
    details = parsed.get("details", {})
    label = _c("FOUND", GREEN) if found else _c("not found", GREY)
    print(f"\n  {_c('Parser result:', BOLD)} {label}")

    for key, value in details.items():
        if not value or key == "raw":
            continue
        if isinstance(value, list):
            preview = ", ".join(str(v)[:80] for v in value[:5])
            print(f"    {_c(key + ':', GREY)} {preview}")
        else:
            print(f"    {_c(key + ':', GREY)} {str(value)[:120]}")


def _prompt_user(prompt: str, default: str = "y") -> str:
    try:
        answer = input(prompt).strip()
        return answer if answer else default
    except (EOFError, KeyboardInterrupt):
        return "n"


MAX_STEPS = 30


def run_agent(
    policy: RLPolicy,
    session,
    auto: bool = False,
    max_steps: int = MAX_STEPS,
    log_dir: str = "logs",
) -> None:
    logger = StepLogger(log_dir=log_dir)
    executor = CommandExecutor(session=session, log_path=os.path.join(log_dir, "execution.jsonl"))
    state = HostState()

    print(_c(f"\n[*] Session log: {logger.path}", GREY))
    print(_c("[*] Starting agent loop. Press Ctrl+C to abort.\n", GREY))

    for step in range(1, max_steps + 1):
        print(f"\n{_c(f'Step {step}/{max_steps}', BOLD)}")
        _print_state(state)

        if state.current_privilege == 1:
            print(_c("\n[OK] Root access achieved. Stopping.", GREEN))
            break

        state_vec = state.to_vector()
        action = policy.predict(state_vec)
        top_actions = policy.top_actions(state_vec, n=3)
        _print_action_suggestion(action, top_actions)

        if action == Action.STOP:
            print(_c("\n[*] Policy suggests STOP.", YELLOW))
            if auto or _prompt_user("  Stop? [Y/n]: ", "y").lower() in ("y", ""):
                break
            action = _manual_action_select()

        if not auto:
            answer = _prompt_user("\n  Execute? [Y/n/s(skip)/q(quit)]: ", "y").lower()
            if answer == "q":
                print(_c("[*] Aborted by operator.", YELLOW))
                break
            if answer == "s":
                print(_c("[*] Skipping action.", GREY))
                continue
            if answer not in ("y", ""):
                action = _manual_action_select()

        commands = get_commands(action)
        if not commands and action != Action.STOP:
            print(_c("  [!] No commands mapped to this action (decision-only action).", YELLOW))
            combined_output = ""
        else:
            combined_output = ""
            for cmd in commands:
                print(f"\n  {_c('$', CYAN)} {cmd}")
                result = executor.execute(cmd, action_name=action.name)
                if result["blocked"]:
                    print(_c(f"  [BLOCKED] {result['error']}", RED))
                    continue
                if result["error"]:
                    print(_c(f"  [ERROR] {result['error']}", RED))
                out = result["output"]
                combined_output += out + "\n"
                lines = out.splitlines()
                for line in lines[:25]:
                    print(f"  {_c('|', GREY)} {line}")
                if len(lines) > 25:
                    print(_c(f"  | ... ({len(lines)-25} more lines)", GREY))

        state_before = state.summary()
        parsed = parse_output(action, combined_output)
        _print_parsed_result(parsed)
        update_state(state, action, parsed)

        logger.log(
            {
                "step": step,
                "state_before": state_before,
                "action": action.name,
                "commands": commands,
                "parsed": parsed,
                "state_after": state.summary(),
            }
        )

        if action == Action.VERIFY_ROOT and state.current_privilege == 1:
            print(_c("\n[OK] Root confirmed.", GREEN))
            break

    print(_c(f"\n[*] Agent loop complete. Log: {logger.path}", GREY))
    _print_state(state)


def _manual_action_select() -> Action:
    print("\n  Available actions:")
    for action in Action:
        print(f"    [{action.value:>2d}] {action.name}")
    while True:
        raw = _prompt_user("  Select action ID: ", str(Action.STOP.value))
        try:
            return Action(int(raw))
        except (ValueError, KeyError):
            print("  Invalid choice. Try again.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="postex-agent",
        description="RL-guided Linux post-exploitation decision support agent.",
    )
    parser.add_argument("--session", choices=["manual", "metasploit", "ssh"], default="manual")
    parser.add_argument("--model-path", default="artifacts/dqn_model.pt")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--device", default=None)

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default=None)
    parser.add_argument("--key", default=None)
    parser.add_argument("--port", type=int, default=22)

    parser.add_argument("--msf-session", type=int, default=1)
    parser.add_argument("--msf-host", default="127.0.0.1")
    parser.add_argument("--msf-port", type=int, default=55553)
    parser.add_argument("--msf-password", default="msf")

    parser.add_argument("--shell", default=None)
    return parser


def main() -> None:
    _print_banner()
    parser = _build_parser()
    args = parser.parse_args()

    print(_c(f"[*] Loading RL policy from: {args.model_path}", CYAN))
    try:
        policy = RLPolicy(model_path=args.model_path, device=args.device)
        print(_c("[OK] Policy loaded.", GREEN))
    except FileNotFoundError as exc:
        print(_c(f"[!] {exc}", RED))
        print(_c("[*] Train first: python -m postex_agent.rl.train_dqn --episodes 10000", YELLOW))
        raise SystemExit(1)

    try:
        session = _build_session(args)
    except Exception as exc:
        print(_c(f"[!] Session error: {exc}", RED))
        raise SystemExit(1)

    try:
        run_agent(
            policy=policy,
            session=session,
            auto=args.auto,
            max_steps=args.max_steps,
            log_dir=args.log_dir,
        )
    except KeyboardInterrupt:
        print(_c("\n[*] Interrupted by user.", YELLOW))
    finally:
        session.close()


if __name__ == "__main__":
    main()

