"""Common contract every Worker Agent must implement.

Per architecture.md (Orchestration Layer / Worker Agents Layer):
- Agents are independent modules, not a fixed pipeline.
- The Supervisor discovers and calls agents dynamically through
  `agent_registry.py`.
- Every agent receives the shared `graph_state` dict, does its job, and
  returns an updated copy of that dict. Agents never write to Storage and
  never call each other directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict


class BaseAgent(ABC):
    """Abstract base class for all Worker Agents.

    Subclasses must set `name` (used as the registry key and as the node
    name inside the LangGraph workflow) and should set `description` for
    documentation / catalog purposes (see Documentation/agent_catalog.md).
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent against the shared workflow state.

        Contract:
        - MUST NOT mutate `state` in place; return a new dict (or a shallow
          copy with updated keys) so Orchestration's state_manager can diff
          / checkpoint reliably.
        - MUST NOT raise on expected/operational failures (bad input file,
          missing field, downstream service error). Those should be
          reported back through `state["errors"]` and an agent-specific
          status key (e.g. `ocr_status`) so the Supervisor's routing_logic
          can decide the next step. Only truly unexpected programming
          errors should propagate.
        - MUST NOT write to Storage directly and MUST NOT call another
          Agent directly; only Orchestration does that.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"<{self.__class__.__name__} name={self.name!r}>"
