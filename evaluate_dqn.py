from __future__ import annotations

"""
Compatibility wrapper.

Canonical implementation lives in `postex_agent.rl.evaluate_dqn`.
"""

from postex_agent.rl.evaluate_dqn import evaluate as evaluate_trained_agent
from postex_agent.rl.evaluate_dqn import parse_seeds, print_report
from postex_agent.rl.evaluate_dqn import _cli as main


if __name__ == "__main__":
    main()

