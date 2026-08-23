"""Summary agent: question + RAG context → grounded notes for writer_agent."""

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
    """Orchestration stage: read request/rag → write state['summary']."""

    name = "summary_agent"
    description = "Summarize retrieved legal context into grounded notes for writer_agent."

    def __init__(
        self,
        client: SummaryClient | None = None,
        config: SummaryConfig | None = None,
    ) -> None:
        self.config = config or SummaryConfig.from_env()
        self.client = client or SummaryClient(self.config)

    def summarize(self, question: Any, rag_result: Any) -> SummaryAgentResult:
        """Standalone entry: (question, RAGResult) → SummaryAgentResult."""
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
        """Orchestration entry: reads request/rag(_result), writes state['summary']."""
        if not isinstance(state, dict):
            raise TypeError("SummaryAgent.run expects GraphState as a dict")

        updated = dict(state)
        question = self._resolve_question(updated)
        rag_result = self._adapt_rag(updated)

        # RAG stage skipped / absent → empty notes (non-blocking for the pipeline).
        err = rag_result.get("error") if isinstance(rag_result.get("error"), dict) else {}
        if err.get("code") == "missing_rag":
            updated["summary"] = {"success": True, "rag_summary_text": ""}
            return updated

        result = self.summarize(question, rag_result)
        updated["summary"] = self._to_state_summary(result)
        return updated

    @staticmethod
    def _resolve_question(state: Dict[str, Any]) -> str:
        request = state.get("request") if isinstance(state.get("request"), dict) else {}
        return (
            request.get("question")
            or state.get("question")
            or state.get("accompanying_text")
            or request.get("accompanying_text")
            or ""
        )

    @staticmethod
    def _adapt_rag(state: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize state['rag'] / state['rag_result'] into RAGResult shape."""
        block = state.get("rag") if isinstance(state.get("rag"), dict) else None
        if not block:
            block = state.get("rag_result") if isinstance(state.get("rag_result"), dict) else {}
        block = block or {}

        # Pipeline slot: {success, rag_data, error?}
        if "rag_data" in block:
            return {
                "success": block.get("success", False),
                "data": block.get("rag_data"),
                "error": block.get("error"),
            }

        # Bare retrieval payload: {operation, query, results}
        if "results" in block:
            return {"success": True, "data": block, "error": block.get("error")}

        # rag_agent simplified: {context, sources, result_count}
        context = (block.get("context") or "").strip()
        if context:
            query = state.get("rag_query") or SummaryAgent._resolve_question(state) or ""
            return {
                "success": True,
                "data": {
                    "operation": "retrieve",
                    "query": query,
                    "results": [{"chunk_id": "rag-context-0", "text": context}],
                },
                "error": block.get("error"),
            }

        # Already RAGResult: {success, data, error?}
        if "data" in block or "success" in block:
            return {
                "success": block.get("success", bool(block.get("data"))),
                "data": block.get("data"),
                "error": block.get("error"),
            }

        return {
            "success": False,
            "data": None,
            "error": block.get("error") or {"code": "missing_rag", "message": "No RAG context in state."},
        }

    @staticmethod
    def _to_state_summary(result: SummaryAgentResult) -> Dict[str, Any]:
        """Map SummaryAgentResult → state['summary'] for writer/routing."""
        if result.success and result.data is not None:
            return {"success": True, "rag_summary_text": result.data.summary}
        return {
            "success": False,
            "rag_summary_text": None,
            "error": result.error or {"code": "unknown_error", "message": "Summary generation failed."},
        }
