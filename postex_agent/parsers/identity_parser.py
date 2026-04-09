"""Parsers for OS and user identification outputs."""
from __future__ import annotations

import re
from typing import Optional

from postex_agent.parsers.base_parser import BaseParser


_KERNEL_RE = re.compile(r"\b(\d+\.\d+\.\d+(?:[-._a-zA-Z0-9]+)?)\b")
_PRETTY_NAME_RE = re.compile(r'^PRETTY_NAME="?([^"\n]+)"?', re.MULTILINE)
_UID_RE = re.compile(r"uid=(\d+)\(([^)]+)\)")
_GROUPS_RE = re.compile(r"groups=([^\n]+)")

_CONTAINER_MARKERS = {
    "docker",
    "containerd",
    "kubepods",
    "podman",
    "libpod",
    "lxc",
}

_PRIVILEGED_GROUPS = {
    "sudo",
    "wheel",
    "adm",
    "docker",
    "lxd",
    "libvirt",
    "disk",
    "shadow",
    "root",
}


def _extract_group_names(raw: str) -> list[str]:
    match = _GROUPS_RE.search(raw)
    if not match:
        return []

    groups_text = match.group(1).strip()
    groups: list[str] = []
    for entry in groups_text.split(","):
        stripped = entry.strip()
        name_match = re.search(r"\(([^)]+)\)", stripped)
        name = name_match.group(1) if name_match else stripped
        if name and name not in groups:
            groups.append(name)
    return groups


def _extract_container_indicators(raw: str) -> list[str]:
    lowered = raw.lower()
    indicators: list[str] = []

    for marker in _CONTAINER_MARKERS:
        if marker in lowered and marker not in indicators:
            indicators.append(marker)

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("container:"):
            value = stripped.split(":", 1)[1].strip().lower()
            if value and value not in indicators:
                indicators.append(value)

    return indicators


class OSParser(BaseParser):
    def parse(self, raw_output: str) -> dict:
        raw = raw_output or ""
        kernel_match = _KERNEL_RE.search(raw)
        pretty_match = _PRETTY_NAME_RE.search(raw)

        kernel_version: Optional[str] = kernel_match.group(1) if kernel_match else None
        os_name: Optional[str] = pretty_match.group(1).strip() if pretty_match else None
        if os_name is None and "Linux" in raw:
            os_name = "Linux"

        container_indicators = _extract_container_indicators(raw)

        return {
            "vector_found": bool(kernel_version or os_name),
            "exploitable": False,
            "details": {
                "raw": raw,
                "kernel_version": kernel_version,
                "os_name": os_name,
                "container_indicators": container_indicators,
                "is_containerized": bool(container_indicators),
            },
        }


class UserParser(BaseParser):
    def parse(self, raw_output: str) -> dict:
        raw = raw_output or ""
        uid_match = _UID_RE.search(raw)

        uid = int(uid_match.group(1)) if uid_match else None
        username = uid_match.group(2) if uid_match else None

        if username is None:
            whoami = raw.strip().splitlines()
            if len(whoami) == 1 and whoami[0]:
                username = whoami[0].strip()
                uid = 0 if username == "root" else uid

        groups = _extract_group_names(raw)
        privileged_groups = [group for group in groups if group.lower() in _PRIVILEGED_GROUPS]

        is_root = bool(uid == 0 or username == "root" or "uid=0(" in raw)

        return {
            "vector_found": bool(username is not None or uid is not None),
            "exploitable": is_root,
            "details": {
                "raw": raw,
                "uid": uid,
                "username": username,
                "groups": groups,
                "privileged_groups": privileged_groups,
                "is_root": is_root,
            },
        }
