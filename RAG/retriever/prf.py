"""Pseudo Relevance Feedback — expand query from top retrieved passages."""

from __future__ import annotations

from collections import Counter
from typing import List, Sequence

from RAG.configuration.rag_config_loader import prf_config
from RAG.retriever.text_utils import tokenize
from RAG.vector_store.vector_store_interface import SearchResult


def extract_expansion_terms(
    results: Sequence[SearchResult],
    *,
    original_query: str,
    max_terms: int | None = None,
) -> List[str]:
    max_terms = max_terms or prf_config.max_expand_terms
    query_terms = set(tokenize(original_query, min_len=prf_config.min_term_len, drop_stopwords=True))
    counts: Counter[str] = Counter()
    for doc in results[: prf_config.top_n_docs]:
        counts.update(tokenize(doc["text"], min_len=prf_config.min_term_len, drop_stopwords=True))

    out: List[str] = []
    for term, _ in counts.most_common(max_terms * 3):
        if term not in query_terms:
            out.append(term)
        if len(out) >= max_terms:
            break
    return out


def expand_query_with_prf(query: str, results: Sequence[SearchResult]) -> str:
    if not prf_config.enabled or not results:
        return query
    terms = extract_expansion_terms(results, original_query=query)
    return f"{query.strip()} {' '.join(terms)}".strip() if terms else query
