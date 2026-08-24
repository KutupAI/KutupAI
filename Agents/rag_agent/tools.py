"""
rag_agent tools — only shared-service integration allowed here.

Calls RAG/client (never Storage, never ChromaDB directly).
Heavy RAG deps are imported lazily so Agent registration stays lightweight.
"""

from typing import Any, Dict, Mapping, Optional


def run_layer_retrieve(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Retrieve-only Layers_contracts path used by Orchestration.

    Fills ``state["rag"]`` via ``handle_layer_state`` (no LLM answer).
    """
    from RAG.client.contract_adapter import handle_layer_state

    return handle_layer_state(state)


def search_legal_context(query: str, top_k: Optional[int] = None):
    """Legacy helper: retrieve Mevzuat / regulation passages for a raw query."""
    from RAG.client import RetrievalRequest
    from RAG.client.rag_client import get_legal_context

    return get_legal_context(RetrievalRequest(query=query, top_k=top_k))


def search_legal_context_as_dict(
    query: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Same as search_legal_context, returned as a plain dict for graph_state."""
    response = search_legal_context(query=query, top_k=top_k)
    return {
        "context": response.context,
        "sources": response.sources,
        "result_count": response.result_count,
    }
