"""
base_agent.py
---------------
Contract every Worker Agent must implement.

Supervisor calls run(state) in-process; Agents never write to Storage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Independent worker unit selected dynamically by the Supervisor."""

    name: str = "base_agent"
    description: str = ""

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute this agent's responsibility and return the updated graph state.

        Args:
            state: Shared Orchestration graph_state (document text, partial results, ...).

        Returns:
            The same state dict with this agent's outputs merged in.
        """
