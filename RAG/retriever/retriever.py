"""
retriever.py
--------------
نقطة الدخول للبحث: تحويل query لمتجه، ثم البحث في مخزن المتجهات وإرجاع
أفضل top_k نتائج. المرحلة الأولى: Top-K بسيط بدون BM25 أو Reranking.
"""

from typing import Optional

from RAG.configuration.rag_config_loader import retrieval_config
from RAG.embeddings.embedding_model import embed_text
from RAG.vector_store.chroma_store import get_vector_store
from RAG.vector_store.vector_store_interface import SearchResult


def retrieve(query: str, top_k: Optional[int] = None) -> list[SearchResult]:
    """
    استرجاع أقرب top_k مقاطع قانونية لسؤال المستخدم.

    Args:
        query: سؤال المستخدم (بالتركية عادةً).
        top_k: عدد النتائج المطلوبة. إذا لم يُحدَّد، يُستخدم القيمة
               الافتراضية من rag_config.yaml.

    Returns:
        قائمة SearchResult مرتبة حسب درجة التشابه تنازليًا.
    """
    effective_top_k = top_k or retrieval_config.default_top_k
    effective_top_k = min(effective_top_k, retrieval_config.max_top_k)

    query_embedding = embed_text(query)

    store = get_vector_store()
    results = store.search(query_embedding=query_embedding, top_k=effective_top_k)

    return results
