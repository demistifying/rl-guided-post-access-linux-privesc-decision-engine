"""Registry for action-specific output parsers."""
from __future__ import annotations

from typing import Dict

from postex_agent.core.actions import Action
from postex_agent.parsers.base_parser import BaseParser
from postex_agent.parsers.capability_parser import CapabilityParser
from postex_agent.parsers.cron_parser import CronParser
from postex_agent.parsers.credential_parser import CredentialParser
from postex_agent.parsers.identity_parser import OSParser, UserParser
from postex_agent.parsers.kernel_parser import KernelParser
from postex_agent.parsers.sudo_parser import SudoParser
from postex_agent.parsers.suid_parser import SuidParser
from postex_agent.parsers.writable_parser import WritableParser


_PARSERS: Dict[Action, BaseParser] = {
    Action.IDENTIFY_OS: OSParser(),
    Action.IDENTIFY_USER: UserParser(),
    Action.CHECK_SUDO: SudoParser(),
    Action.CHECK_SUID: SuidParser(),
    Action.CHECK_CAPABILITIES: CapabilityParser(),
    Action.CHECK_WRITABLE: WritableParser(),
    Action.CHECK_CRON: CronParser(),
    Action.SEARCH_CREDENTIALS: CredentialParser(),
    Action.CHECK_KERNEL: KernelParser(),
    Action.VERIFY_ROOT: UserParser(),
}


def get_parser(action: Action) -> BaseParser:
    return _PARSERS.get(action, BaseParser())


def parse_output(action: Action, output: str) -> dict:
    parser = get_parser(action)
    return parser.parse(output)

