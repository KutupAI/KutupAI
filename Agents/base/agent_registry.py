"""Dynamic registry so the Supervisor can pick agents without hardcoding
imports (architecture.md, principle #3 and #7: Agents are discoverable and
new agents can be added without touching Orchestration Core).

Usage in an agent module:

    from Agents.base.agent_registry import register
    from Agents.base.base_agent import BaseAgent

    @register
    class MyAgent(BaseAgent):
        name = "my_agent"
        ...

Usage in Orchestration:

    from Agents.base.agent_registry import get_agent, list_agents

    agent_cls = get_agent("ocr_agent")
    new_state = agent_cls().run(state)
"""

from __future__ import annotations

from typing import Dict, Type

from Agents.base.base_agent import BaseAgent

_REGISTRY: Dict[str, Type[BaseAgent]] = {}


class AgentRegistryError(Exception):
    """Raised for registry misuse (duplicate/unknown agent name)."""


def register(agent_cls: Type[BaseAgent]) -> Type[BaseAgent]:
    """Class decorator that registers a BaseAgent subclass by its `name`."""
    if not issubclass(agent_cls, BaseAgent):
        raise AgentRegistryError(
            f"{agent_cls!r} must subclass Agents.base.base_agent.BaseAgent"
        )
    name = getattr(agent_cls, "name", "") or ""
    if not name:
        raise AgentRegistryError(
            f"{agent_cls.__name__} must define a non-empty 'name' before registration."
        )
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not agent_cls:
        raise AgentRegistryError(
            f"Agent name '{name}' is already registered to {existing.__name__}."
        )
    _REGISTRY[name] = agent_cls
    return agent_cls


def get_agent(name: str) -> Type[BaseAgent]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise AgentRegistryError(
            f"No agent registered under name '{name}'. Known agents: {sorted(_REGISTRY)}"
        ) from exc


def list_agents() -> Dict[str, Type[BaseAgent]]:
    """Return a read-only snapshot of the registry (name -> agent class)."""
    return dict(_REGISTRY)


def unregister(name: str) -> None:
    """Mainly for tests: remove an agent from the registry."""
    _REGISTRY.pop(name, None)
