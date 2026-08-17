"""Deterministic routing for the interactive legal-answering pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from RAG.retriever.query_metadata import get_query_metadata_extractor
from RAG.retriever.text_utils import fold_turkish


@dataclass(frozen=True)
class QueryPlan:
    name: str
    mode: str
    use_prf: bool
    use_reranker: bool
    use_graph: bool | str | None
    rationale: str


def choose_query_plan(question: str) -> QueryPlan:
    """Choose the smallest transparent pipeline that fits the question."""
    metadata = get_query_metadata_extractor().extract(question)
    normalized = fold_turkish(question).casefold()
    if metadata.get("law_number") and metadata.get("article_no"):
        return QueryPlan("exact_citation", "vector", False, True, None,
                         "Açık kanun ve madde bulundu: metadata filtresi + hassas sıralama.")
    article_lookup_terms = ("hangi maddede", "hangi madde", "maddesi nedir", "hangi maddes")
    if any(term in normalized for term in article_lookup_terms):
        return QueryPlan("article_lookup", "hybrid", False, True, False,
                         "Madde bulma sorusu: hibrit arama ile başlık ve metin eşleştirmesi.")
    graph_terms = ("baglant", "iliski", "atıf", "atif", "ilgili madde", "karsilastir", "karşılaştır")
    if any(term in normalized for term in graph_terms):
        return QueryPlan("legal_relationship", "hybrid", True, True, "full",
                         "İlişki/atıf sorusu: hybrid + PRF + reranker + Graph-RAG FULL.")
    keyword_terms = ("teblig", "yonetmelik", "yönetmelik", "resmi gazete", "kavram", "tanım", "tanimi")
    if any(term in normalized for term in keyword_terms):
        return QueryPlan("lexical_legal_lookup", "hybrid", False, True, False,
                         "Anahtar terim ağırlıklı hukuk sorgusu: hibrit lexical + semantic arama.")
    return QueryPlan("semantic_fast", "vector", False, True, False,
                     "Genel anlam tabanlı soru: hızlı dense retrieval + reranker.")
