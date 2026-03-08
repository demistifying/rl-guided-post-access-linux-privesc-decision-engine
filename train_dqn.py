from __future__ import annotations

"""
Compatibility wrapper.

Canonical implementation lives in `postex_agent.rl.train_dqn`.
"""

from postex_agent.rl.train_dqn import train as train_agent
from postex_agent.rl.train_dqn import _cli as main


if __name__ == "__main__":
    main()

