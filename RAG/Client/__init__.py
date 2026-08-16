"""Agents API: RetrievalRequest → get_legal_context()."""

from RAG.client.rag_client import get_legal_context
from RAG.client.retrieval_request import RetrievalRequest
from RAG.client.retrieval_response import RetrievalResponse

__all__ = ["RetrievalRequest", "RetrievalResponse", "get_legal_context"]
