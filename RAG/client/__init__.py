"""
KutupAI - RAG Client Package Initialization
-------------------------------------------
يصدّر الواجهة البرمجية والكلاسات اللازمة للـ Agents وسكربتات الاختبار.
"""

from RAG.client.rag_client import get_legal_context
from RAG.client.retrieval_request import RetrievalRequest
from RAG.client.retrieval_response import RetrievalResponse

# `from RAG.client import *` kullanımında dışa açılan istemci sözleşmesini sınırlar.
__all__ = [
    "get_legal_context",
    "RetrievalRequest",
    "RetrievalResponse",
]
