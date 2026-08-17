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
    # Hukukî cross-encoder yararlıdır ancak güçlü Hybrid+PRF sinyalini silmemelidir.
    # Sıra-normalize füzyon, farklı logit ölçekli modeller arasında kararlıdır.
    reranker_order = sorted(range(len(candidates)), key=lambda index: float(scores[index]), reverse=True)
    reranker_rank = {index: rank for rank, index in enumerate(reranker_order, start=1)}
    count = max(1, len(candidates))
    base_weight = reranker_config.base_rank_weight
    ranked = []
    for index, item in enumerate(candidates):
        base_score = 1.0 - (index / count)
        cross_score = 1.0 - ((reranker_rank[index] - 1) / count)
        combined = (base_weight * base_score) + ((1.0 - base_weight) * cross_score)
        ranked.append((item, float(scores[index]), combined, index + 1, reranker_rank[index]))
    ranked.sort(key=lambda entry: (entry[2], entry[1]), reverse=True)

    out: List[SearchResult] = []
    for item, score, combined, base_rank, cross_rank in ranked[:top_k]:
        meta = dict(item["metadata"])
        meta["cross_encoder_score"] = round(float(score), 6)
        meta["hybrid_rank"] = base_rank
        meta["reranker_rank"] = cross_rank
        meta["blended_score"] = round(float(combined), 6)
        out.append(
            SearchResult(
                id=item["id"],
                text=item["text"],
                metadata=meta,
                score=round(float(score), 6),
            )
        )
    return out
