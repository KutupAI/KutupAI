"""
KutupAI - Gelişmiş Sorgu Genişletme Stratejileri (Advanced Query Expansion)
---------------------------------------------------------------------------
none / synonym / prf / smart_rule stratejilerini destekler.
"""

from __future__ import annotations

from typing import Dict, List
import re

from RAG.configuration.rag_config_loader import query_expansion_config
from RAG.retriever.hybrid import hybrid_search
from RAG.retriever.prf import expand_query_with_prf

# Eş anlamlı sözlüğü.
_SYNONYMS: Dict[str, List[str]] = {
    "ihbar": ["bildirim", "fesih bildirimi"],
    "fesih": ["sona erdirme", "işten çıkarma"],
    "işçi": ["çalışan", "personel"],
    "işveren": ["patron", "şirket"],
    "kıdem": ["çalışma süresi"],
    "izin": ["yıllık izin"],
    "yedek subay": ["askerlik süresi", "muvazzaf"],
    "tazminat": ["nakdi tazminat", "idari tazminat"],
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


# Kural tabanlı genişletme.
def expand_rule_based(query: str) -> List[str]:
    expanded_queries = [query]
    article_match = re.search(r"(\d+)\.\s*maddes", query, re.IGNORECASE)
    if not article_match:
        article_match = re.search(r"(?:Madde|madde|MADDE|m\.)\s+(\d+)", query)
        
    if article_match:
        article_no = article_match.group(1)
        expanded_queries.append(f"Madde {article_no}")
            
    return expanded_queries


# Genişletme stratejisi.
def apply_strategy(query: str, strategy: str | None = None) -> str:
    strategy = (strategy or query_expansion_config.selected_strategy or "none").lower()
    
    if strategy in ("none", "off", ""):
        return query
        
    if strategy == "synonym":
        return expand_synonyms(query)
        
    if strategy == "prf":
        return expand_query_with_prf(query, hybrid_search(query, top_k=10))
        
    if strategy == "synonym+prf":
        syn_query = expand_synonyms(query)
        return expand_query_with_prf(syn_query, hybrid_search(syn_query, top_k=10))
        
    if strategy == "smart_rule":
        rule_expanded = expand_rule_based(query)
        base_query = rule_expanded[-1] if len(rule_expanded) > 1 else query
        return expand_synonyms(base_query)

    return query


def get_expanded_queries(query: str, strategy: str | None = None) -> List[str]:
    """Tek bir sorgu yerine hibrit aramada kullanılacak sorgu varyasyonları listesi."""
    strategy = (strategy or query_expansion_config.selected_strategy or "none").lower()
    queries = [query]
    
    rule_queries = expand_rule_based(query)
    for q in rule_queries:
        if q not in queries:
            queries.append(q)
            
    syn_query = expand_synonyms(query)
    if syn_query != query and syn_query not in queries:
        queries.append(syn_query)
        
    return queries
