"""Track 3 host archetypes for realistic training distributions.

The simulator samples one hidden host archetype per episode, then derives
correlated vector-presence probabilities, richness means, and exploit-success
parameters from that archetype.  This keeps the live agent unchanged while
making the training world substantially closer to real post-exploitation
decision making.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from postex_agent.core.state import VECTOR_KEYS


DEFAULT_DETERMINISTIC_SUCCESS: Dict[str, float] = {
    "sudo": 0.98,
    "suid": 0.95,
    "capabilities": 0.90,
    "cron": 0.55,
    "kernel": 0.50,
}


@dataclass(frozen=True)
class ArchetypeTemplate:
    name: str
    weight: float
    vector_probs: Dict[str, float]
    richness_means: Dict[str, float]
    kernel_success_range: Tuple[float, float]
    cap_success_base: float = 0.90
    cron_base_success: float = 0.55
    cron_chain_success: float = 0.85
    cred_quality_levels: Tuple[Tuple[float, float], ...] = (
        (0.33, 0.50),
        (0.66, 0.30),
        (1.00, 0.20),
    )


@dataclass(frozen=True)
class SampledArchetypeProfile:
    name: str
    vector_probs: Dict[str, float]
    richness_means: Dict[str, float]
    success_probs: Dict[str, float]
    cron_chain_success: float
    cred_quality_levels: Tuple[Tuple[float, float], ...]


ARCHETYPE_TEMPLATES: List[ArchetypeTemplate] = [
    ArchetypeTemplate(
        name="neglected_server",
        weight=0.35,
        vector_probs={
            "sudo": 0.55,
            "suid": 0.60,
            "capabilities": 0.25,
            "writable_path": 0.60,
            "cron": 0.45,
            "credentials": 0.50,
            "kernel": 0.20,
        },
        richness_means={
            "sudo": 3.0,
            "suid": 6.0,
            "capabilities": 2.5,
            "writable_path": 6.0,
            "cron": 4.0,
            "credentials": 4.0,
            "kernel": 1.5,
        },
        kernel_success_range=(0.50, 0.60),
        cap_success_base=0.92,
        cron_base_success=0.60,
        cron_chain_success=0.88,
        cred_quality_levels=((0.33, 0.25), (0.66, 0.45), (1.00, 0.30)),
    ),
    ArchetypeTemplate(
        name="corporate_app",
        weight=0.25,
        vector_probs={
            "sudo": 0.25,
            "suid": 0.30,
            "capabilities": 0.15,
            "writable_path": 0.20,
            "cron": 0.25,
            "credentials": 0.35,
            "kernel": 0.08,
        },
        richness_means={
            "sudo": 1.5,
            "suid": 3.0,
            "capabilities": 1.5,
            "writable_path": 2.0,
            "cron": 2.5,
            "credentials": 3.0,
            "kernel": 1.0,
        },
        kernel_success_range=(0.40, 0.50),
        cap_success_base=0.90,
        cron_base_success=0.55,
        cron_chain_success=0.84,
        cred_quality_levels=((0.33, 0.35), (0.66, 0.45), (1.00, 0.20)),
    ),
    ArchetypeTemplate(
        name="containerized_app",
        weight=0.15,
        vector_probs={
            "sudo": 0.05,
            "suid": 0.12,
            "capabilities": 0.35,
            "writable_path": 0.45,
            "cron": 0.05,
            "credentials": 0.25,
            "kernel": 0.02,
        },
        richness_means={
            "sudo": 1.0,
            "suid": 2.0,
            "capabilities": 4.0,
            "writable_path": 5.0,
            "cron": 1.0,
            "credentials": 2.5,
            "kernel": 1.0,
        },
        kernel_success_range=(0.35, 0.45),
        cap_success_base=0.93,
        cron_base_success=0.50,
        cron_chain_success=0.80,
        cred_quality_levels=((0.33, 0.25), (0.66, 0.55), (1.00, 0.20)),
    ),
    ArchetypeTemplate(
        name="hardened_host",
        weight=0.15,
        vector_probs={
            "sudo": 0.10,
            "suid": 0.10,
            "capabilities": 0.05,
            "writable_path": 0.10,
            "cron": 0.08,
            "credentials": 0.10,
            "kernel": 0.04,
        },
        richness_means={
            "sudo": 1.0,
            "suid": 1.5,
            "capabilities": 1.0,
            "writable_path": 1.0,
            "cron": 1.0,
            "credentials": 1.5,
            "kernel": 1.0,
        },
        kernel_success_range=(0.35, 0.42),
        cap_success_base=0.88,
        cron_base_success=0.45,
        cron_chain_success=0.72,
        cred_quality_levels=((0.33, 0.60), (0.66, 0.30), (1.00, 0.10)),
    ),
    ArchetypeTemplate(
        name="ctf_lab",
        weight=0.10,
        vector_probs={key: 0.10 for key in VECTOR_KEYS},
        richness_means={
            "sudo": 2.0,
            "suid": 3.0,
            "capabilities": 2.0,
            "writable_path": 3.0,
            "cron": 2.0,
            "credentials": 2.0,
            "kernel": 1.0,
        },
        kernel_success_range=(0.50, 0.65),
        cap_success_base=0.92,
        cron_base_success=0.58,
        cron_chain_success=0.88,
        cred_quality_levels=((0.33, 0.15), (0.66, 0.35), (1.00, 0.50)),
    ),
]


ARCHETYPE_NAMES: Tuple[str, ...] = tuple(template.name for template in ARCHETYPE_TEMPLATES)


def get_archetype_template(name: str) -> ArchetypeTemplate:
    for template in ARCHETYPE_TEMPLATES:
        if template.name == name:
            return template
    raise KeyError(f"Unknown host archetype: {name}")


def _sample_ctf_profile(template: ArchetypeTemplate, rng: random.Random) -> SampledArchetypeProfile:
    vector_probs = {key: 0.08 for key in VECTOR_KEYS}
    richness_means = {key: 1.0 for key in VECTOR_KEYS}

    dominant_choices = ["sudo", "suid", "capabilities", "cron", "kernel"]
    dominant = rng.choice(dominant_choices)
    backup_choices = [value for value in dominant_choices if value != dominant]
    backup = rng.choice(backup_choices)

    vector_probs[dominant] = 0.78
    vector_probs[backup] = 0.32
    richness_means[dominant] = 6.0 if dominant != "kernel" else 2.0
    richness_means[backup] = 3.5 if backup != "kernel" else 1.5

    # Keep supporting evidence visible when cron is the intended path.
    if dominant == "cron" or backup == "cron":
        vector_probs["writable_path"] = max(vector_probs["writable_path"], 0.65)
        richness_means["writable_path"] = 5.0

    # CTF boxes frequently hand the user a clue-bearing credential file.
    if dominant == "sudo":
        vector_probs["credentials"] = max(vector_probs["credentials"], 0.18)
        richness_means["credentials"] = max(richness_means["credentials"], 2.5)

    success_probs = dict(DEFAULT_DETERMINISTIC_SUCCESS)
    success_probs["capabilities"] = 0.92
    success_probs["cron"] = 0.58
    success_probs["kernel"] = rng.uniform(*template.kernel_success_range)

    return SampledArchetypeProfile(
        name=template.name,
        vector_probs=vector_probs,
        richness_means=richness_means,
        success_probs=success_probs,
        cron_chain_success=template.cron_chain_success,
        cred_quality_levels=template.cred_quality_levels,
    )


def sample_archetype_profile(
    rng: random.Random,
    force_name: str | None = None,
) -> SampledArchetypeProfile:
    """Sample a correlated host profile for one episode.

    `force_name` is used by tests and deterministic debugging.
    """
    if force_name is not None:
        template = get_archetype_template(force_name)
    else:
        weights = [template.weight for template in ARCHETYPE_TEMPLATES]
        template = rng.choices(ARCHETYPE_TEMPLATES, weights=weights, k=1)[0]

    if template.name == "ctf_lab":
        return _sample_ctf_profile(template, rng)

    success_probs = dict(DEFAULT_DETERMINISTIC_SUCCESS)
    success_probs["capabilities"] = template.cap_success_base
    success_probs["cron"] = template.cron_base_success
    success_probs["kernel"] = rng.uniform(*template.kernel_success_range)

    return SampledArchetypeProfile(
        name=template.name,
        vector_probs=dict(template.vector_probs),
        richness_means=dict(template.richness_means),
        success_probs=success_probs,
        cron_chain_success=template.cron_chain_success,
        cred_quality_levels=template.cred_quality_levels,
    )
