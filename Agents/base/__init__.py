"""Shared base contract + dynamic registry for all Worker Agents."""

from Agents.base.agent_registry import (
    AgentRegistryError,
    get_agent,
    list_agents,
    register,
    unregister,
)
from Agents.base.base_agent import BaseAgent

__all__ = [
    "BaseAgent",
    "register",
    "get_agent",
    "list_agents",
    "unregister",
    "AgentRegistryError",
]
