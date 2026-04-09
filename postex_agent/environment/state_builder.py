"""Translate parsed command output into HostState updates.

Populates all 35 state dimensions:
  - Original 17: privilege, os/user id, checked/found flags
  - Richness[7]: item counts per vector (normalised)
  - Exploit failures[7]: incremented on failed exploit actions
  - Credential quality: cred_count, cred_quality
  - Temporal: time_step, cumulative_risk (set by caller via update_temporal)
"""
from __future__ import annotations

from typing import Dict

from postex_agent.core.actions import Action, VECTOR_BY_CHECK_ACTION, VECTOR_BY_EXPLOIT_ACTION
from postex_agent.core.state import HostState, MAX_RICHNESS

# Risk penalties per vector — must match simulation_env for consistency
VECTOR_RISK_PENALTIES: Dict[str, float] = {
    "sudo":         0.10,
    "suid":         0.20,
    "capabilities": 0.30,
    "writable_path": 0.25,
    "cron":         0.18,
    "credentials":  0.22,
    "kernel":       0.45,
}

MAX_CUMULATIVE_RISK = sum(VECTOR_RISK_PENALTIES.values())


def _dedupe(existing: list[str], incoming: list[str]) -> list[str]:
    merged = list(existing)
    for value in incoming:
        if value not in merged:
            merged.append(value)
    return merged


def _item_count_for_vector(vector: str, details: dict) -> int:
    """Extract the relevant item count from parser details for richness."""
    if vector == "sudo":
        return len(details.get("sudo_commands", []))
    if vector == "suid":
        return len(details.get("exploitable_bins", []))
    if vector == "capabilities":
        return len(details.get("exploitable_binaries", []))
    if vector == "writable_path":
        return len(details.get("writable_paths", []))
    if vector == "cron":
        return len(details.get("cron_jobs", []))
    if vector == "credentials":
        return int(details.get("cred_count", len(details.get("credentials", []))))
    if vector == "kernel":
        return len(details.get("known_cves", []))
    return 0


def update_state(state: HostState, action: Action, parsed: Dict) -> HostState:
    """Update HostState from a parsed action result.

    This populates the original 17 binary dims AND the new richness,
    credential quality, and exploit failure dimensions.
    """
    details = parsed.get("details", {}) if isinstance(parsed, dict) else {}
    vector_found = bool(parsed.get("vector_found", False)) if isinstance(parsed, dict) else False

    if action == Action.IDENTIFY_OS:
        state.os_identified = True
        kernel_version = details.get("kernel_version")
        os_name = details.get("os_name")
        container_indicators = details.get("container_indicators", [])
        if kernel_version:
            state.kernel_version = kernel_version
        if os_name:
            state.os_info = os_name
        if container_indicators:
            state.container_indicators = _dedupe(
                state.container_indicators,
                container_indicators,
            )
        state.is_containerized = bool(
            state.container_indicators or details.get("is_containerized", False)
        )
        return state

    if action == Action.IDENTIFY_USER:
        state.user_identified = True
        username = details.get("username")
        is_root = bool(details.get("is_root", False))
        groups = details.get("groups", [])
        privileged_groups = details.get("privileged_groups", [])
        if username:
            state.current_user = username
        if groups:
            state.user_groups = _dedupe(state.user_groups, groups)
        if privileged_groups:
            state.privileged_groups = _dedupe(state.privileged_groups, privileged_groups)
        if is_root:
            state.current_privilege = 1
        return state

    if action in VECTOR_BY_CHECK_ACTION:
        vector = VECTOR_BY_CHECK_ACTION[action]
        state.checked[vector] = True
        if vector_found:
            state.found[vector] = True

        # ── Populate richness (normalised item count) ─────────────
        count = _item_count_for_vector(vector, details)
        state.richness[vector] = min(count / MAX_RICHNESS, 1.0)

        # ── Populate rich metadata (for CLI display) ──────────────
        if vector == "sudo":
            state.sudo_commands = _dedupe(state.sudo_commands, details.get("sudo_commands", []))
            state.sudo_nopasswd_entries = _dedupe(
                state.sudo_nopasswd_entries,
                details.get("nopasswd_entries", []),
            )
            state.sudo_password_entries = _dedupe(
                state.sudo_password_entries,
                details.get("passworded_entries", []),
            )
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
            state.cron_writable_targets = _dedupe(
                state.cron_writable_targets,
                details.get("writable_targets", details.get("potentially_writable", [])),
            )
        elif vector == "credentials":
            state.credentials_found = _dedupe(
                state.credentials_found, details.get("credentials", [])
            )
            # ── Credential quality signals ────────────────────────
            cred_count = details.get("cred_count", len(details.get("credentials", [])))
            cred_quality = details.get("cred_quality", 0.0)
            state.cred_count = min(cred_count / MAX_RICHNESS, 1.0)
            state.cred_quality = max(state.cred_quality, cred_quality)  # keep best
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
        achieved_root = details.get("is_root", False) is True
        if achieved_root:
            state.current_privilege = 1
        else:
            # ── Exploit did not achieve root — increment failure counter ──
            state.exploit_failures[vector] = state.exploit_failures.get(vector, 0) + 1
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


def update_temporal(
    state: HostState,
    step: int,
    max_steps: int,
    cumulative_risk: float,
) -> None:
    """Update the temporal/risk features each step.

    Called by agent_cli and real_env after every action.
    """
    state.time_step = step / max_steps
    state.cumulative_risk = min(cumulative_risk / MAX_CUMULATIVE_RISK, 1.0)
