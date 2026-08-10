"""CrossEncoder re-ranking."""

from __future__ import annotations

from functools import lru_cache
from typing import List

from RAG.configuration.rag_config_loader import reranker_config, runtime_config
from RAG.vector_store.vector_store_interface import SearchResult


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(reranker_config.model_name, device=runtime_config.device)


def rerank(query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
    if not results:
        return []
    if not reranker_config.enabled:
        return results[:top_k]

    candidates = results[: max(top_k, reranker_config.top_n)]
    scores = _model().predict([(query, item["text"]) for item in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)

    out: List[SearchResult] = []
    for item, score in ranked[:top_k]:
        meta = dict(item["metadata"])
        meta["cross_encoder_score"] = round(float(score), 6)
        out.append(
            SearchResult(
                id=item["id"],
                text=item["text"],
                metadata=meta,
                score=round(float(score), 6),
            )
        )
    return out
