"""Retrieval pipeline: expand → hybrid → PRF → CrossEncoder."""

from __future__ import annotations

from typing import List, Optional

from RAG.configuration.rag_config_loader import (
    prf_config,
    query_expansion_config,
    retrieval_config,
)
from RAG.retriever.hybrid import hybrid_search
from RAG.retriever.prf import expand_query_with_prf
from RAG.retriever.query_expansion import apply_strategy
from RAG.retriever.reranker import rerank
from RAG.vector_store.vector_store_interface import SearchResult


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    *,
    mode: Optional[str] = None,
    use_prf: Optional[bool] = None,
    use_reranker: Optional[bool] = None,
    expansion_strategy: Optional[str] = None,
    where: Optional[dict] = None,
) -> List[SearchResult]:
    if not query or not str(query).strip():
        return []

    k = min(max(top_k or retrieval_config.default_top_k, 1), retrieval_config.max_top_k)
    candidate_k = min(max(k * 3, retrieval_config.candidate_k), retrieval_config.max_top_k)

    strategy = expansion_strategy
    if strategy is None and query_expansion_config.enabled:
        strategy = query_expansion_config.selected_strategy
    q = apply_strategy(query.strip(), strategy) if strategy else query.strip()

    hits = hybrid_search(q, top_k=candidate_k, where=where, mode=mode)

    if (prf_config.enabled if use_prf is None else use_prf) and hits:
        prf_q = expand_query_with_prf(q, hits)
        if prf_q != q:
            hits = hybrid_search(prf_q, top_k=candidate_k, where=where, mode=mode)
            q = prf_q

    if use_reranker is False:
        return hits[:k]
    return rerank(query=q, results=hits, top_k=k)
