"""
End-to-end CLI test using a MockSession.

Tests the full pipeline:
  model loads → policy selects → commands generated
  → mock output returned → parser extracts data
  → state builder populates 35 dims → policy selects next action
  → agent terminates correctly
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Dict

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from postex_agent.core.actions import Action
from postex_agent.core.state import STATE_DIM
from postex_agent.sessions.base_session import BaseSession
from postex_agent.cli.agent_cli import run_agent
from postex_agent.rl.policy_inference import RLPolicy

RESET   = "\033[0m"
GREEN   = "\033[32m"
RED     = "\033[31m"
CYAN    = "\033[36m"
YELLOW  = "\033[33m"
BOLD    = "\033[1m"


# ── Mock Linux command outputs ───────────────────────────────────────────────

MOCK_OUTPUTS: Dict[str, str] = {
    # IDENTIFY_OS commands
    "uname -a": (
        "Linux vulnbox 5.15.0-56-generic #62-Ubuntu SMP "
        "Tue Jan 24 14:03:05 UTC 2023 x86_64 GNU/Linux"
    ),
    "cat /etc/os-release 2>/dev/null": (
        'PRETTY_NAME="Ubuntu 22.04.1 LTS"\n'
        'NAME="Ubuntu"\n'
        'VERSION_ID="22.04"\n'
        'ID=ubuntu\n'
    ),

    # IDENTIFY_USER commands
    "id": "uid=33(www-data) gid=33(www-data) groups=33(www-data)",
    "whoami": "www-data",

    # CHECK_SUDO
    "sudo -l 2>/dev/null": (
        "Matching Defaults entries for www-data on vulnbox:\n"
        "    env_reset, mail_badpass\n"
        "\n"
        "User www-data may run the following commands on vulnbox:\n"
        "    (ALL : ALL) NOPASSWD: /usr/bin/find\n"
        "    (ALL : ALL) NOPASSWD: /usr/bin/vim\n"
    ),

    # CHECK_SUID
    "find / -perm -4000 -type f 2>/dev/null": (
        "/usr/bin/passwd\n"
        "/usr/bin/su\n"
        "/usr/bin/find\n"
        "/usr/bin/python3\n"
        "/usr/bin/mount\n"
        "/usr/bin/umount\n"
    ),

    # CHECK_CAPABILITIES
    "getcap -r / 2>/dev/null": (
        "/usr/bin/python3 = cap_setuid+ep\n"
        "/usr/bin/ping = cap_net_raw+ep\n"
    ),

    # CHECK_WRITABLE
    "find / -xdev -type d -perm -0002 2>/dev/null | head -n 200": (
        "/tmp\n"
        "/var/tmp\n"
        "/dev/shm\n"
    ),

    # CHECK_CRON
    "cat /etc/crontab 2>/dev/null": (
        "SHELL=/bin/sh\n"
        "PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin\n"
        "\n"
        "*/5 * * * * root /opt/scripts/cleanup.sh\n"
        "0 * * * * root /opt/scripts/backup.py\n"
    ),
    "ls -la /etc/cron* 2>/dev/null": "/etc/crontab\n",
    "crontab -l 2>/dev/null": "no crontab for www-data\n",

    # SEARCH_CREDENTIALS
    "env | grep -iE 'pass|token|key|secret' 2>/dev/null": (
        "DB_PASSWORD=supersecret123\n"
        "API_TOKEN=tok_live_abc123def456\n"
    ),
    "grep -RniE 'password|passwd|token|secret|apikey' /etc /opt /home 2>/dev/null | head -n 200": (
        "/opt/app/config.ini:password=admin_plaintext_pass\n"
        "/home/backup/.ssh/id_rsa:-----BEGIN RSA PRIVATE KEY-----\n"
    ),

    # CHECK_KERNEL
    "uname -r": "5.15.0-56-generic",
    "cat /proc/version 2>/dev/null": (
        "Linux version 5.15.0-56-generic (buildd@lcy02-amd64-006) "
        "(gcc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0, GNU ld (GNU Binutils for Ubuntu) 2.38)"
    ),

    # Exploit commands — simulate success by returning root id
    "sudo /usr/bin/find . -exec /bin/sh \\; -quit": "",
}

# After a sudo exploit, id/whoami return root
MOCK_ROOT_OUTPUTS: Dict[str, str] = {
    "id": "uid=0(root) gid=0(root) groups=0(root)",
    "whoami": "root",
}


class MockSession(BaseSession):
    """Simulates a Linux target by returning pre-scripted outputs.

    Tracks state: after a successful exploit, switches to root outputs.
    """

    def __init__(self):
        self._is_root = False
        self._commands_run: list[str] = []

    def run(self, command: str, timeout: int = 30) -> dict:
        self._commands_run.append(command)
        cmd = command.strip()

        # If we've "become root", return root identity for id/whoami
        if self._is_root and cmd in MOCK_ROOT_OUTPUTS:
            return {
                "output": MOCK_ROOT_OUTPUTS[cmd],
                "error": "",
                "exit_code": 0,
            }

        # Detect exploit commands — sudo, suid, or cap python3 escalation
        is_exploit = (
            (cmd.startswith("sudo ") and any(b in cmd for b in ["find", "vim", "python"]))
            or ("os.setuid(0)" in cmd)
            or ("os.execl" in cmd)
            or (cmd.endswith("-p") and any(b in cmd for b in ["/bin/sh", "/bin/bash"]))
        )
        if is_exploit:
            self._is_root = True
            return {"output": "", "error": "", "exit_code": 0}

        # Look up in mock outputs
        output = MOCK_OUTPUTS.get(cmd, "")
        return {
            "output": output,
            "error": "",
            "exit_code": 0,
        }

    def close(self) -> None:
        pass

    @property
    def commands_run(self) -> list[str]:
        return list(self._commands_run)


# ── Test runner ──────────────────────────────────────────────────────────────

def run_e2e_test():
    print(f"\n{'=' * 70}")
    print(f"  {BOLD}END-TO-END CLI TEST{RESET}")
    print(f"{'=' * 70}\n")

    results = []

    def _check(name: str, condition: bool, detail: str = ""):
        status = f"{GREEN}PASS{RESET}" if condition else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {name}")
        if detail:
            print(f"         {CYAN}{detail}{RESET}")
        results.append((name, condition))

    # ── 1. Model loads ────────────────────────────────────────────────────
    model_path = "artifacts/dqn_model.pt"
    try:
        policy = RLPolicy(model_path=model_path)
        _check("Model loads from checkpoint", True,
               f"state_dim={policy._agent.config.state_dim}")
    except Exception as e:
        _check("Model loads from checkpoint", False, str(e))
        print(f"\n  {RED}Cannot continue without model.{RESET}")
        return False

    _check("State dim matches", policy._agent.config.state_dim == STATE_DIM,
           f"expected={STATE_DIM}, got={policy._agent.config.state_dim}")

    # ── 2. Run agent loop with mock session ───────────────────────────────
    print(f"\n  {YELLOW}Running agent loop with MockSession...{RESET}\n")

    session = MockSession()
    log_dir = os.path.join(tempfile.gettempdir(), "postex_e2e_test")
    # Clean previous runs
    if os.path.exists(log_dir):
        for f in os.listdir(log_dir):
            os.remove(os.path.join(log_dir, f))
    os.makedirs(log_dir, exist_ok=True)

    try:
        run_agent(
            policy=policy,
            session=session,
            auto=True,         # non-interactive
            max_steps=20,
            log_dir=log_dir,
        )
        _check("Agent loop completed without crash", True)
    except Exception as e:
        _check("Agent loop completed without crash", False, str(e))
        return False

    # ── 3. Verify commands were actually executed ─────────────────────────
    cmds = session.commands_run
    _check("Commands were executed on session", len(cmds) > 0,
           f"{len(cmds)} commands run")

    # ── 4. Verify session log was written ─────────────────────────────────
    log_files = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("session_")],
        reverse=True,  # latest first
    )
    _check("Session log file created", len(log_files) > 0,
           f"log files: {log_files}")

    if log_files:
        log_path = os.path.join(log_dir, log_files[0])
        with open(log_path, "r") as f:
            log_entries = [json.loads(line) for line in f if line.strip()]

        _check("Log entries written", len(log_entries) > 0,
               f"{len(log_entries)} entries")

        # ── 5. Verify state progression ───────────────────────────────────
        first = log_entries[0]
        last  = log_entries[-1]
        _check("First log entry has action field", "action" in first,
               f"action={first.get('action', '?')}")
        _check("Last log entry has state_after", "state_after" in last,
               f"state_after={str(last.get('state_after', '?'))[:80]}")

        # Check if agent terminated correctly.
        # In mock testing, three outcomes are acceptable:
        #   1. Root achieved (would happen with a more complete mock)
        #   2. Agent chose STOP (exhausted all exploit vectors)
        #   3. Last logged action is an exploit (STOP executes on next loop but
        #      isn't logged because the loop breaks first in auto mode)
        root_achieved = last.get("state_after", {}).get("current_privilege") == 1
        last_action = last.get("action", "")
        all_exploits_tried = all(
            any(e.get("action") == a for e in log_entries)
            for a in ["EXPLOIT_SUDO", "EXPLOIT_SUID", "EXPLOIT_CAP",
                       "EXPLOIT_CRON", "EXPLOIT_KERNEL"]
        )
        graceful = root_achieved or last_action == "STOP" or all_exploits_tried
        _check("Agent terminates gracefully", graceful,
               f"root={root_achieved}, last={last_action}, all_exploits_tried={all_exploits_tried}")

    # ── 6. Verify action variety ──────────────────────────────────────────
    actions_taken = set()
    if log_files:
        for entry in log_entries:
            actions_taken.add(entry.get("action", ""))

    _check("Multiple distinct actions taken", len(actions_taken) >= 2,
           f"actions: {sorted(actions_taken)}")

    has_enum = any(a.startswith("CHECK_") or a.startswith("IDENTIFY_") for a in actions_taken)
    _check("Enumeration actions were taken", has_enum)

    has_exploit = any(a.startswith("EXPLOIT_") for a in actions_taken)
    _check("Exploit actions were taken", has_exploit)

    # ── Summary ───────────────────────────────────────────────────────────
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    total  = len(results)

    print(f"\n{'=' * 70}")
    color = GREEN if failed == 0 else RED
    print(f"  {color}{BOLD}Results: {passed} passed, {failed} failed out of {total} checks{RESET}")
    if failed == 0:
        print(f"  {GREEN}All E2E checks passed!{RESET}")
    print(f"{'=' * 70}\n")

    return failed == 0


if __name__ == "__main__":
    success = run_e2e_test()
    sys.exit(0 if success else 1)
