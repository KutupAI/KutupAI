"""Query expansion strategies (none / synonym / prf)."""

from __future__ import annotations

from typing import Dict, List

from RAG.configuration.rag_config_loader import query_expansion_config
from RAG.retriever.hybrid import hybrid_search
from RAG.retriever.prf import expand_query_with_prf

_SYNONYMS: Dict[str, List[str]] = {
    "ihbar": ["bildirim", "fesih bildirimi"],
    "fesih": ["sona erdirme", "işten çıkarma"],
    "işçi": ["çalışan", "personel"],
    "işveren": ["patron", "şirket"],
    "kıdem": ["çalışma süresi"],
    "izin": ["yıllık izin"],
}


def expand_synonyms(query: str, max_extra: int | None = None) -> str:
    max_extra = max_extra or query_expansion_config.max_extra_terms
    extra: List[str] = []
    lower = query.lower()
    for key, values in _SYNONYMS.items():
        if key not in lower:
            continue
        for value in values:
            if value.lower() not in lower and value not in extra:
                extra.append(value)
            if len(extra) >= max_extra:
                return f"{query.strip()} {' '.join(extra)}"
    return f"{query.strip()} {' '.join(extra)}".strip() if extra else query


def apply_strategy(query: str, strategy: str | None = None) -> str:
    strategy = (strategy or query_expansion_config.selected_strategy or "none").lower()
    if strategy in ("none", "off", ""):
        return query
    if strategy == "synonym":
        return expand_synonyms(query)
    if strategy == "prf":
        return expand_query_with_prf(query, hybrid_search(query, top_k=10))
    if strategy == "synonym+prf":
        return expand_query_with_prf(expand_synonyms(query), hybrid_search(expand_synonyms(query), top_k=10))
    return query
