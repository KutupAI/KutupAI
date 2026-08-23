"""
writer_agent tools -- optional legal grounding via the RAG client.

NOT called by WriterAgent.run() by default: the standard flow already
receives grounded context through state["summary"]["rag_summary_text"]
(produced upstream by the RAG/Summary agents), so calling RAG again
here would duplicate work and tokens. This helper is kept available
for a future/optional path where Writer Agent needs to pull additional
legal citations beyond what Summary already provides.

Heavy RAG deps are imported lazily so importing writer_agent does not
require the RAG package unless this function is actually called.
"""

from typing import Any, Dict, Optional


def fetch_citation_context(
    query: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Pull legal citations the writer may reference in the official reply.

    Optional utility -- not part of the default WriterAgent.run() path.
    """
    from RAG.client import RetrievalRequest
    from RAG.client.rag_client import get_legal_context

    response = get_legal_context(RetrievalRequest(query=query, top_k=top_k))
    return {
        "context": response.context,
        "sources": response.sources,
        "result_count": response.result_count,
    }
