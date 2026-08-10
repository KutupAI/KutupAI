"""
agent_registry.py
-------------------
Dynamic discovery table for the Supervisor.

Adding a new Agent = implement BaseAgent + register() here.
Orchestration core does not change.
"""

from __future__ import annotations

from typing import Dict, List, Type

from Agents.base.base_agent import BaseAgent

_REGISTRY: Dict[str, Type[BaseAgent]] = {}


def register(agent_cls: Type[BaseAgent]) -> Type[BaseAgent]:
    """Class decorator / helper that registers an Agent by its `name`."""
    key = getattr(agent_cls, "name", None) or agent_cls.__name__
    if not key:
        raise ValueError(f"Agent class {agent_cls!r} must define a non-empty name")
    _REGISTRY[key] = agent_cls
    return agent_cls


def get_agent(name: str) -> BaseAgent:
    """Instantiate a registered Agent by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown agent '{name}'. Registered: {list_agents()}")
    return _REGISTRY[name]()


def list_agents() -> List[str]:
    """Names of all registered Agents (order is registration order)."""
    return list(_REGISTRY.keys())


def clear_registry() -> None:
    """Test helper — wipe registrations."""
    _REGISTRY.clear()
