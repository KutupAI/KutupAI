"""Tools exposed to SummaryAgent (Inference integration stays behind SummaryClient)."""

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
    """Run summarization and return a plain dict for graph_state['summary_result']."""
    agent = SummaryAgent(client=client, config=config)
    return agent.summarize(question, rag_result).model_dump()
