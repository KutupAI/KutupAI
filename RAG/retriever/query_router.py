"""Deterministic routing for the interactive legal-answering pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from RAG.retriever.query_metadata import get_query_metadata_extractor
from RAG.retriever.query_frame import build_query_frame
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
    """Sorunun kanıt yapısına göre güvenli retrieval yolunu seçer."""
    frame = build_query_frame(question, extractor=get_query_metadata_extractor())
    metadata = {
        "law_number": frame.intent.primary_law_number,
        "article_no": frame.article_numbers[0] if frame.article_numbers else None,
    }
    intent = frame.intent
    normalized = fold_turkish(question).casefold()
    if frame.needs_amendment_evidence:
        return QueryPlan("amendment_lookup", "hybrid", True, True, False,
                         "Değişiklik/iptal/yürürlük sorusu: kanun retrieval + değişiklik cetveli.")
    if frame.kind == "comparison":
        return QueryPlan("comparison_lookup", "hybrid", True, True, False,
                         "Karşılaştırma sorusu: her hukukî taraf için dengeli kanıt havuzu.")
    if frame.kind == "multi_law_relation":
        return QueryPlan("multi_law_relation", "hybrid", True, True, "full",
                         "Birden fazla kanun/atıf: her kanun için ayrı aday havuzu + Graph-RAG.")
    if metadata.get("law_number") and metadata.get("article_no"):
        return QueryPlan("exact_citation", "vector", False, True, None,
                         "Açık kanun ve madde bulundu: metadata filtresi + hassas sıralama.")
    if intent.needs_multiple_evidence:
        return QueryPlan("multi_article_same_law", "hybrid", True, True, "full",
                         "Birden fazla hüküm isteniyor: alt kanıtlar birlikte getirilir.")
    if intent.kind == "temporal":
        return QueryPlan("temporal_lookup", "hybrid", False, True, False,
                         "Tarih/yürürlük sorusu: tarih içeren hüküm ve cetvel önceliği.")
    if intent.kind == "authority":
        return QueryPlan("authority_lookup", "hybrid", False, True, False,
                         "Yetki sorusu: kurum ve görev hükmü önceliği.")
    if intent.kind == "sanction":
        return QueryPlan("sanction_lookup", "hybrid", False, True, False,
                         "Ceza/tutar sorusu: fiil ve yaptırım hükmü birlikte aranır.")
    if intent.kind == "condition":
        return QueryPlan("condition_lookup", "hybrid", False, True, False,
                         "Şart/koşul sorusu: ilgili bentlerin tamamı aranır.")
    article_lookup_terms = ("hangi maddede", "hangi madde", "maddesi nedir", "hangi maddes")
    if any(term in normalized for term in article_lookup_terms):
        return QueryPlan("article_lookup", "hybrid", False, True, False,
                         "Madde bulma sorusu: hibrit arama ile başlık ve metin eşleştirmesi.")
    graph_terms = ("baglant", "iliski", "atıf", "atif", "ilgili madde", "karsilastir", "karşılaştır")
    if any(term in normalized for term in graph_terms):
        return QueryPlan("legal_relationship", "hybrid", True, True, "full",
                         "İlişki/atıf sorusu: hybrid + PRF + reranker + Graph-RAG FULL.")
    # Ceza miktarı, beyanname ve usulsüzlük soruları çoğu zaman hem hukuki
    # terimi hem de sayısal/tarifeye bağlı bir sonucu içerir. Bu nedenle tek
    # başına dense arama yerine lexical ve semantic sinyalleri birleştirilir.
    factual_legal_terms = (
        "usulsuzluk", "beyanname", "mukellef", "ceza tutari", "ceza miktari",
        "para cezasi", "idari para", "guncel ceza", "tarife", "ceza haddi",
    )
    keyword_terms = (
        "teblig", "yonetmelik", "resmi gazete", "kavram", "tanim", "tanimi",
    )
    if any(term in normalized for term in factual_legal_terms + keyword_terms):
        return QueryPlan("lexical_legal_lookup", "hybrid", False, True, False,
                         "Anahtar terim/tutar ağırlıklı hukuk sorgusu: hibrit lexical + semantic arama.")
    # Serbest biçimli hukuk soruları önceden tanımlı anahtar kelimelerle
    # sınırlı değildir. Varsayılan hybrid yol, yeni ifade biçimlerinde BM25'in
    # terim eşleşmesini ve vector aramanın anlamsal eşleşmesini birlikte kullanır.
    return QueryPlan("semantic_hybrid", "hybrid", False, True, False,
                     "Serbest biçimli hukuk sorusu: hybrid arama + hassas reranker.")
