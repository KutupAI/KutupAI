"""
Routing Agent
=============

Production-grade hybrid routing agent for the KutupAI pipeline.

Determines, for an incoming document, the correct:
    institution -> presidency/general directorate -> department -> unit ->
    authority/makam -> processing channel

using a hybrid of rule-based, keyword, BM25, semantic-similarity, metadata,
legal-authority, entity-matching and (optionally) LLM-reasoning scores,
followed by hierarchy resolution, authority resolution, conflict detection,
ambiguity detection and confidence calibration.

See agent.py for the orchestrator and README-level documentation in the
module docstrings of each file.
"""

from .agent import RoutingAgent, process, get_default_agent
from .models import (
    SharedStateInput,
    RoutingResult,
    Route,
    RouteCandidate,
    RoutingStatus,
    ConfidenceLevel,
    ProcessingChannel,
)
from .knowledge_base import KnowledgeBase, default_knowledge_base

__all__ = [
    "RoutingAgent",
    "process",
    "get_default_agent",
    "SharedStateInput",
    "RoutingResult",
    "Route",
    "RouteCandidate",
    "RoutingStatus",
    "ConfidenceLevel",
    "ProcessingChannel",
    "KnowledgeBase",
    "default_knowledge_base",
]
