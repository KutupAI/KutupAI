"""Hybrid retrieval: BM25 + vector search fused with RRF."""

from __future__ import annotations

from typing import Dict, List, Optional

from RAG.configuration.rag_config_loader import retrieval_config
from RAG.retriever.bm25_index import get_bm25_index
from RAG.vector_store.chroma_store import get_vector_store
from RAG.vector_store.vector_store_interface import SearchResult


def _rrf_fuse(
    lists: List[List[SearchResult]],
    weights: List[float],
    rrf_k: int,
    top_k: int,
) -> List[SearchResult]:
    scores: Dict[str, float] = {}
    payload: Dict[str, SearchResult] = {}
    for results, weight in zip(lists, weights):
        for rank, item in enumerate(results, start=1):
            key = item["id"] or str(hash(item["text"]))
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (rrf_k + rank))
            if key not in payload or item["score"] > payload[key]["score"]:
                payload[key] = item

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        SearchResult(
            id=payload[key]["id"],
            text=payload[key]["text"],
            metadata={**payload[key]["metadata"], "rrf_score": round(score, 6)},
            score=round(score, 6),
        )
        for key, score in ordered
    ]


def _filter(results: List[SearchResult], where: Optional[dict]) -> List[SearchResult]:
    if not where:
        return results
    return [r for r in results if all(r["metadata"].get(k) == v for k, v in where.items())]


def hybrid_search(
    query: str,
    top_k: int,
    *,
    where: Optional[dict] = None,
    mode: Optional[str] = None,
) -> List[SearchResult]:
    mode = (mode or retrieval_config.mode).lower()
    k = max(top_k, retrieval_config.candidate_k)

    if mode == "vector":
        return get_vector_store().similarity_search(query, k, where=where)[:top_k]
    if mode == "bm25":
        return _filter(get_bm25_index().search(query, k * 3 if where else k), where)[:top_k]

    vec = get_vector_store().similarity_search(query, k, where=where)
    lex = _filter(get_bm25_index().search(query, k * 3 if where else k), where)
    return _rrf_fuse(
        [vec, lex],
        [retrieval_config.vector_weight, retrieval_config.bm25_weight],
        retrieval_config.rrf_k,
        top_k,
    )
