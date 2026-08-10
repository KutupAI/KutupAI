"""
Worker Agents Layer

Independent modules discovered via `Agents.base.agent_registry`.
No Agent writes to Storage — results return through graph_state only.

## Adding an Agent

1. Create `Agents/<name>_agent/` with `agent.py`, `prompts.py`, `tools.py`, `config.py`
2. Subclass `BaseAgent` and implement `run(state)`
3. Decorate with `@register` from `agent_registry`
4. Import the module from `Agents/__init__.py` so registration runs at startup
"""

from Agents.rag_agent import agent as _rag_agent  # noqa: F401
from Agents.ocr_agent import agent as _ocr_agent  # noqa: F401

__all__ = ["_rag_agent", "_ocr_agent"]
