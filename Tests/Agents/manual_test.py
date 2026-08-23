"""Run SummaryAgent against the live Inference server.

From the repository root:
    python Tests/Agents/manual_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Agents.summary_agent.agent import SummaryAgent
from Agents.summary_agent.mock_data import MOCK_STATE


def main() -> None:
    updated_state = SummaryAgent().run(MOCK_STATE)
    print(json.dumps(updated_state.get("summary"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
