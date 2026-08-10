"""
rag_agent
---------
Finds laws/regulations relevant to the document. Calls: RAG then Inference.
"""

from __future__ import annotations

from typing import Any, Dict

from Agents.base.agent_registry import register
from Agents.base.base_agent import BaseAgent
from Agents.rag_agent.tools import search_legal_context_as_dict


@register
class RagAgent(BaseAgent):
    name = "rag_agent"
    description = "Retrieve relevant Mevzuat and regulations for the document content."

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = (
            state.get("rag_query")
            or state.get("document_text")
            or state.get("summary")
            or ""
        )
        if not str(query).strip():
            state["rag_result"] = {
                "context": "",
                "sources": [],
                "result_count": 0,
                "error": "empty_query",
            }
            return state

        state["rag_result"] = search_legal_context_as_dict(query=str(query))
        return state
