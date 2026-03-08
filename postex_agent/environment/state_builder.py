"""Translate parsed command output into HostState updates."""
from __future__ import annotations

from typing import Dict

from postex_agent.core.actions import Action, VECTOR_BY_CHECK_ACTION, VECTOR_BY_EXPLOIT_ACTION
from postex_agent.core.state import HostState


def _dedupe(existing: list[str], incoming: list[str]) -> list[str]:
    merged = list(existing)
    for value in incoming:
        if value not in merged:
            merged.append(value)
    return merged


def update_state(state: HostState, action: Action, parsed: Dict) -> HostState:
    details = parsed.get("details", {}) if isinstance(parsed, dict) else {}
    vector_found = bool(parsed.get("vector_found", False)) if isinstance(parsed, dict) else False

    if action == Action.IDENTIFY_OS:
        state.os_identified = True
        kernel_version = details.get("kernel_version")
        os_name = details.get("os_name")
        if kernel_version:
            state.kernel_version = kernel_version
        if os_name:
            state.os_info = os_name
        return state

    if action == Action.IDENTIFY_USER:
        state.user_identified = True
        username = details.get("username")
        is_root = bool(details.get("is_root", False))
        if username:
            state.current_user = username
        if is_root:
            state.current_privilege = 1
        return state

    if action in VECTOR_BY_CHECK_ACTION:
        vector = VECTOR_BY_CHECK_ACTION[action]
        state.checked[vector] = True
        if vector_found:
            state.found[vector] = True

        if vector == "sudo":
            state.sudo_commands = _dedupe(state.sudo_commands, details.get("sudo_commands", []))
        elif vector == "suid":
            state.exploitable_suid_bins = _dedupe(
                state.exploitable_suid_bins, details.get("exploitable_bins", [])
            )
        elif vector == "capabilities":
            state.exploitable_caps = _dedupe(
                state.exploitable_caps, details.get("exploitable_binaries", [])
            )
        elif vector == "writable_path":
            state.writable_paths = _dedupe(state.writable_paths, details.get("writable_paths", []))
        elif vector == "cron":
            state.cron_jobs = _dedupe(state.cron_jobs, details.get("cron_jobs", []))
        elif vector == "credentials":
            state.credentials_found = _dedupe(
                state.credentials_found, details.get("credentials", [])
            )
        elif vector == "kernel":
            kernel_version = details.get("kernel_version")
            if kernel_version:
                state.kernel_version = kernel_version
        return state

    if action in VECTOR_BY_EXPLOIT_ACTION:
        vector = VECTOR_BY_EXPLOIT_ACTION[action]
        state.checked[vector] = True
        if vector_found:
            state.found[vector] = True
        if details.get("is_root") is True:
            state.current_privilege = 1
        return state

    if action == Action.VERIFY_ROOT:
        is_root = bool(details.get("is_root", False))
        if is_root:
            state.current_privilege = 1
        username = details.get("username")
        if username:
            state.current_user = username
        return state

    return state

