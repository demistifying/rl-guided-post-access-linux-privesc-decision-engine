"""
Docker-based live integration tests for postex-agent.

Builds vulnerable Linux containers, runs the agent against each via SSH,
and verifies that it correctly achieves root (or correctly stops on
hardened hosts).

Requirements:
  - Docker Desktop running
  - pip install paramiko   (SSH session adapter)

Usage:
  python tests/test_docker_live.py
  python tests/test_docker_live.py --scenario vuln-sudo
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"

DOCKER_DIR = os.path.join(os.path.dirname(__file__), "docker")
BASE_PORT  = 2220

# ── Scenario definitions ─────────────────────────────────────────────────────

SCENARIOS: Dict[str, dict] = {
    "vuln-sudo": {
        "dockerfile_dir": os.path.join(DOCKER_DIR, "vuln-sudo"),
        "expect_root": True,
        "description": "sudo NOPASSWD on /usr/bin/find",
        "expected_vector": "sudo",
    },
    "vuln-suid": {
        "dockerfile_dir": os.path.join(DOCKER_DIR, "vuln-suid"),
        "expect_root": True,
        "description": "SUID bit on python3",
        "expected_vector": "suid",
    },
    "vuln-cron": {
        "dockerfile_dir": os.path.join(DOCKER_DIR, "vuln-cron"),
        "expect_root": False,  # cron exploits are async — agent can't wait
        "description": "writable cron script (detection only)",
        "expected_vector": "cron",
    },
    "vuln-hardened": {
        "dockerfile_dir": os.path.join(DOCKER_DIR, "vuln-hardened"),
        "expect_root": False,
        "description": "no vulnerabilities — agent should STOP",
        "expected_vector": None,
    },
}

SSH_USER     = "testuser"
SSH_PASSWORD = "testpass"


# ── Docker helpers ────────────────────────────────────────────────────────────

def _run(cmd: str, cwd: Optional[str] = None, timeout: int = 120) -> str:
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=cwd, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result.stdout.strip()


def build_image(name: str, dockerfile_dir: str) -> None:
    tag = f"postex-test-{name}"
    print(f"  {CYAN}Building {tag}...{RESET}")
    _run(f'docker build -t {tag} "{dockerfile_dir}"', timeout=300)


def start_container(name: str, port: int) -> str:
    tag = f"postex-test-{name}"
    container_name = f"postex-live-{name}"

    # Remove stale container if exists
    subprocess.run(
        f"docker rm -f {container_name}",
        shell=True, capture_output=True, timeout=10,
    )

    _run(f"docker run -d --name {container_name} -p {port}:22 {tag}")
    print(f"  {CYAN}Started {container_name} on port {port}{RESET}")
    return container_name


def stop_container(container_name: str) -> None:
    subprocess.run(
        f"docker rm -f {container_name}",
        shell=True, capture_output=True, timeout=10,
    )


def wait_for_ssh(port: int, retries: int = 15, delay: float = 2.0) -> bool:
    """Wait until the container's SSH is accepting connections."""
    import socket
    for attempt in range(retries):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=3):
                return True
        except (ConnectionRefusedError, OSError, socket.timeout):
            time.sleep(delay)
    return False


# ── Agent runner ──────────────────────────────────────────────────────────────

def run_agent_against(port: int, report_path: str, log_dir: str) -> dict:
    """Run postex-agent against a container and return results."""
    from postex_agent.cli.agent_cli import run_agent
    from postex_agent.rl.policy_inference import RLPolicy
    from postex_agent.sessions.metasploit_session import SSHSession

    model_path = os.path.join(
        os.path.dirname(__file__), "..", "artifacts", "dqn_model.pt"
    )
    policy = RLPolicy(model_path=model_path)

    session = SSHSession(
        host="127.0.0.1",
        username=SSH_USER,
        password=SSH_PASSWORD,
        port=port,
        timeout=10,
    )

    try:
        run_agent(
            policy=policy,
            session=session,
            auto=True,
            max_steps=20,
            log_dir=log_dir,
            report_out=report_path,
        )
    except Exception as e:
        print(f"  {RED}Agent error: {e}{RESET}")
    finally:
        session.close()

    # Read the session log to extract results
    log_files = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("session_")],
        reverse=True,
    )
    if not log_files:
        return {"steps": 0, "root": False, "actions": [], "error": "No log file"}

    log_path = os.path.join(log_dir, log_files[0])
    entries = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        return {"steps": 0, "root": False, "actions": [], "error": "Empty log"}

    last_state = entries[-1].get("state_after", {})
    actions = [e.get("action", "") for e in entries]
    root = last_state.get("current_privilege") == 1
    found = last_state.get("found", {})

    return {
        "steps": len(entries),
        "root": root,
        "actions": actions,
        "found_vectors": [k for k, v in found.items() if v],
        "final_user": last_state.get("current_user", "?"),
        "report_exists": os.path.exists(report_path),
    }


# ── Test runner ───────────────────────────────────────────────────────────────

def run_scenario(name: str, scenario: dict, port: int) -> bool:
    print(f"\n{'=' * 70}")
    print(f"  {BOLD}SCENARIO: {name}{RESET}")
    print(f"  {scenario['description']}")
    print(f"{'=' * 70}")

    checks: List[tuple] = []

    def _check(label: str, passed: bool, detail: str = ""):
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {label}")
        if detail:
            print(f"         {CYAN}{detail}{RESET}")
        checks.append((label, passed))

    # Build and start
    try:
        build_image(name, scenario["dockerfile_dir"])
        container = start_container(name, port)
    except Exception as e:
        _check("Docker build/start", False, str(e))
        return False

    try:
        # Wait for SSH
        ssh_ok = wait_for_ssh(port)
        _check("SSH reachable", ssh_ok, f"port {port}")
        if not ssh_ok:
            return False

        # Run agent
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "report.md")
            log_dir = os.path.join(tmpdir, "logs")
            os.makedirs(log_dir, exist_ok=True)

            print(f"\n  {YELLOW}Running agent...{RESET}\n")
            results = run_agent_against(port, report_path, log_dir)

        # Verify results
        _check("Agent completed without crash", "error" not in results,
               results.get("error", ""))
        _check("Actions were taken", results["steps"] > 0,
               f"{results['steps']} steps: {results['actions']}")

        if scenario["expect_root"]:
            _check("ROOT achieved", results["root"],
                   f"final_user={results.get('final_user', '?')}")
        else:
            if scenario["expected_vector"]:
                # Should detect the vector even if it can't exploit it
                found = results.get("found_vectors", [])
                _check(f"Detected {scenario['expected_vector']} vector",
                       scenario["expected_vector"] in found,
                       f"found: {found}")
            else:
                # Hardened: should NOT achieve root
                _check("Correctly did NOT escalate", not results["root"],
                       "agent correctly stopped")

        _check("Engagement report generated", results.get("report_exists", False))

    finally:
        stop_container(f"postex-live-{name}")

    passed = all(ok for _, ok in checks)
    return passed


def main():
    parser = argparse.ArgumentParser(description="Docker-based live tests")
    parser.add_argument("--scenario", default=None,
                        help="Run a single scenario (e.g., vuln-sudo)")
    args = parser.parse_args()

    # Check paramiko is available
    try:
        import paramiko  # noqa: F401
    except ImportError:
        print(f"{RED}paramiko is required: pip install paramiko{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  POSTEX-AGENT DOCKER LIVE TESTS{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    scenarios = SCENARIOS
    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"{RED}Unknown scenario: {args.scenario}{RESET}")
            print(f"Available: {', '.join(SCENARIOS)}")
            sys.exit(1)
        scenarios = {args.scenario: SCENARIOS[args.scenario]}

    overall: Dict[str, bool] = {}

    for idx, (name, scenario) in enumerate(scenarios.items()):
        port = BASE_PORT + idx
        passed = run_scenario(name, scenario, port)
        overall[name] = passed

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  {BOLD}SUMMARY{RESET}")
    print(f"{'=' * 70}")
    for name, passed in overall.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {name}: {SCENARIOS[name]['description']}")

    total = len(overall)
    passed_count = sum(1 for v in overall.values() if v)
    color = GREEN if passed_count == total else RED
    print(f"\n  {color}{BOLD}{passed_count}/{total} scenarios passed{RESET}\n")

    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()
