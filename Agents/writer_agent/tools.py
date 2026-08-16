"""
writer_agent tools — legal grounding via RAG client before drafting.

Heavy RAG deps are imported lazily.
"""

from typing import Any, Dict, Optional


def fetch_citation_context(
    query: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Pull legal citations the writer may reference in the official reply."""
    from RAG.client import RetrievalRequest
    from RAG.client.rag_client import get_legal_context

    response = get_legal_context(RetrievalRequest(query=query, top_k=top_k))
    return {
        "context": response.context,
        "sources": response.sources,
        "result_count": response.result_count,
    }
