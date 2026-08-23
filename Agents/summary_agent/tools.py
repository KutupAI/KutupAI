"""Standalone helpers; Orchestration uses SummaryAgent.run(state)."""

from __future__ import annotations

from typing import Any

from .agent import SummaryAgent
from .client import SummaryClient
from .config import SummaryConfig


def summarize_context(
    question: str,
    rag_result: dict[str, Any],
    *,
    config: SummaryConfig | None = None,
    client: SummaryClient | None = None,
) -> dict[str, Any]:
    """(question, RAGResult) → plain dict. Prefer SummaryAgent.run for the pipeline."""
    agent = SummaryAgent(client=client, config=config)
    return agent.summarize(question, rag_result).model_dump()
