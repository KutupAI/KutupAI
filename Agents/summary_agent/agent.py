"""
summary_agent — turns (question, rag_result) into a concise, source-grounded summary.

Flow:
    question + rag_result -> build_prompt -> SummaryClient -> Gemma 3 -> SummaryAgentResult
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ValidationError

from Agents.base.agent_registry import register
from Agents.base.base_agent import BaseAgent

from .client import SummaryClient, SummaryRequest
from .config import SummaryConfig
from .prompts import build_prompt
from .schemas import RAGResult, RAGResultItem, SourceRef, SummaryAgentResult, SummaryData


def _error(code: str, message: str) -> SummaryAgentResult:
    return SummaryAgentResult(success=False, data=None, error={"code": code, "message": message})


@register
class SummaryAgent(BaseAgent):
    """Summarize RAG context for writer_agent. Calls Inference Layer only — no model loading."""

    name = "summary_agent"
    description = "Summarize retrieved legal context into grounded bullet notes for writer_agent."

    def __init__(
        self,
        client: SummaryClient | None = None,
        config: SummaryConfig | None = None,
    ) -> None:
        self.config = config or SummaryConfig.from_env()
        self.client = client or SummaryClient(self.config)

    def summarize(self, question: Any, rag_result: Any) -> SummaryAgentResult:
        """Core entry point — usable directly outside graph state."""

        if not isinstance(question, str) or not question.strip():
            return _error("invalid_input", "`question` must be a non-empty string.")

        try:
            rag = RAGResult.model_validate(rag_result)
        except ValidationError as exc:
            return _error("invalid_input", f"`rag_result` does not match the expected contract: {exc}")

        if not rag.success or rag.data is None:
            reason = (rag.error or {}).get("message", "RAG retrieval failed.") if rag.error else "RAG retrieval failed."
            return _error("rag_failed", reason)

        results: List[RAGResultItem] = rag.data.results
        if not results:
            return _error("empty_context", "RAG returned no results to summarize.")

        prompt = build_prompt(question, results)
        response = self.client.generate(SummaryRequest(prompt=prompt))

        if not response.success:
            return _error("inference_error", response.error or "Inference request failed.")

        summary_text = (response.text or "").strip()
        if not summary_text:
            return _error("invalid_model_response", "Model returned an empty response.")

        if summary_text == self.config.no_context_marker:
            return _error("empty_context", "No retrieved content was relevant to the question.")

        sources = [
            SourceRef(
                chunk_id=r.chunk_id,
                law_number=r.law_number,
                article_no=r.article_no,
                article_type=r.article_type,
                page_start=r.page_start,
                page_end=r.page_end,
            )
            for r in results
        ]

        return SummaryAgentResult(
            success=True,
            data=SummaryData(query=rag.data.query, summary=summary_text, sources=sources),
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestration entry: reads question/rag_result, writes summary_result."""
        updated = dict(state)
        result = self.summarize(updated.get("question"), updated.get("rag_result"))
        updated["summary_result"] = result.model_dump()
        return updated
