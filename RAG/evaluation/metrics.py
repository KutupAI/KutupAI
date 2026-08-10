"""Hit@k and MRR."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    return 1.0 if any(doc_id in relevant_ids for doc_id in ranked_ids[:k]) else 0.0


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def aggregate_metrics(
    per_query_ranked: List[List[str]],
    per_query_relevant: List[Set[str]],
    hit_ks: Sequence[int] = (1, 2, 3),
) -> dict:
    mrr = [reciprocal_rank(r, rel) for r, rel in zip(per_query_ranked, per_query_relevant)]
    metrics = {"MRR": round(mean(mrr), 4), "n_queries": len(per_query_ranked)}
    for k in hit_ks:
        hits = [hit_at_k(r, rel, k) for r, rel in zip(per_query_ranked, per_query_relevant)]
        metrics[f"Hit@{k}"] = round(mean(hits), 4)
    return metrics
