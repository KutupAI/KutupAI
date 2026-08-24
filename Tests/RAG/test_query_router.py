"""Query Router'ın pahalı bağımlılıklar olmadan temel rota testleri."""

from RAG.retriever.query_router import choose_query_plan
from RAG.retriever.query_metadata import QueryIntent


class _EmptyMetadataExtractor:
    """Testte dosya indeksi yerine boş metadata döndürür."""

    def extract(self, _question: str) -> dict:
        return {}

    def extract_intent(self, _question: str) -> QueryIntent:
        """Yeni router'ın beklediği boş ama geçerli soru çerçevesini üretir."""
        return QueryIntent(
            law_numbers=(),
            article_numbers=(),
            primary_law_number=None,
            amending_law_numbers=(),
            kind="general",
            needs_multiple_evidence=False,
        )


def test_penalty_and_declaration_question_uses_hybrid(monkeypatch):
    """Usulsüzlük ve ceza tutarı sorusu hybrid rota ile aranmalıdır."""
    monkeypatch.setattr(
        "RAG.retriever.query_router.get_query_metadata_extractor",
        lambda: _EmptyMetadataExtractor(),
    )
    question = (
        "Bir sermaye şirketinin vergi beyannamesini süresinde vermemesi halinde "
        "hangi derece usulsüzlük sayılır ve güncel ceza tutarı ne kadardır?"
    )

    plan = choose_query_plan(question)

    assert plan.name == "lexical_legal_lookup"
    assert plan.mode == "hybrid"
    assert plan.use_reranker is True
    assert plan.use_prf is False
    assert plan.use_graph is False


def test_unknown_legal_wording_uses_hybrid_by_default(monkeypatch):
    """Önceden tanımlı kelime içermeyen hukuk sorusu vector-only'e düşmemelidir."""
    monkeypatch.setattr(
        "RAG.retriever.query_router.get_query_metadata_extractor",
        lambda: _EmptyMetadataExtractor(),
    )

    plan = choose_query_plan("Bir çalışan ücretini alamazsa hangi hakları olabilir?")

    assert plan.name == "semantic_hybrid"
    assert plan.mode == "hybrid"
    assert plan.use_reranker is True
