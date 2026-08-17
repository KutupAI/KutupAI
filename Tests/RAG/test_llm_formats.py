"""LLM sunucusu gerektirmeyen cevap ve citation sözleşme testleri."""

from __future__ import annotations

import sys
from pathlib import Path


# Pytest, dosyayı doğrudan çalıştırsa bile RAG paketi kökten bulunur.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_legal_answer_serialization_preserves_observability_fields() -> None:
    """UI ve benchmark için gerekli zamanlama/kaynak alanları JSON'a taşınır."""
    from RAG.agent.legal_agent import LegalAnswer

    answer = LegalAnswer(answer="Kaynaklı cevap.", grounded=True, retrieval_ms=12.5, generation_ms=40.0)
    payload = answer.to_dict()

    assert payload["answer"] == "Kaynaklı cevap."
    assert payload["grounded"] is True
    assert payload["retrieval_ms"] == 12.5
    assert LegalAnswer.from_dict(payload) == answer


def test_citation_validator_rejects_labels_not_present_in_context() -> None:
    """Modelin bağlam dışı kaynak uydurması güvenli biçimde reddedilir."""
    from RAG.agent.citations import validate_citations

    sources = [{"label": "S1"}]
    grounded, invalid = validate_citations("Hüküm geçerlidir. [S99]", sources)

    assert grounded is False
    assert invalid == ["S99"]


def test_citation_display_uses_clean_turkish_law_title() -> None:
    """Kaynak satırında dosya adından gelen numara tekrarlanmaz."""
    from RAG.agent.citations import render_citations

    rendered = render_citations([{"label": "S1", "law_number": "5237", "law_name": "5237_Türk Ceza Kanunu", "article_number": "65"}])

    assert rendered == "[S1] 5237 sayılı Türk Ceza Kanunu - Madde 65"


def test_conversation_detects_follow_up_without_repeating_old_query() -> None:
    """Takip işareti, yeni aramayı önceki konuya bağlar; eski metin eklenmez."""
    from RAG.agent.conversation import ConversationTurn, LegalConversation
    from RAG.agent.legal_agent import LegalAnswer

    conversation = object.__new__(LegalConversation)
    previous = ConversationTurn(
        question="CMK 100 maddesinde tutuklama şartları nelerdir?",
        answer=LegalAnswer(answer="Önceki cevap.", grounded=True),
        primary_law_number="5271",
        primary_law_name="Ceza Muhakemesi Kanunu",
    )

    assert conversation._is_related("Peki kaçma şüphesi ne zaman kabul edilir?", previous) is True
    assert conversation._is_related("KVKK 5. maddesi neyi düzenler?", previous) is False


def test_service_and_operational_questions_are_not_sent_to_the_llm() -> None:
    """Yazım hatalı portal ve pratik işlem soruları güvenli sınıfa alınır."""
    from RAG.retriever.query_intent import asks_for_service_lookup

    assert asks_for_service_lookup("Vergi nede sorgulabilir?") is False
    assert asks_for_service_lookup("Vergi borcumu nerden sorgulayabilirim?") is True
    assert asks_for_service_lookup("Arabam çalınırsa ne yapmalıyım?") is False
    assert asks_for_service_lookup("CMK 100. maddede tutuklama şartları nelerdir?") is False


def test_ambiguous_forgiveness_follow_up_requests_clarification() -> None:
    """Kişisel affetmeyi otomatik olarak genel af şeklinde yorumlamaz."""
    from RAG.agent.conversation import LegalConversation

    answer = LegalConversation._clarification_answer("Peki, eğer onu affedersem ne olur?")

    assert answer is not None
    assert answer.grounded is False
    assert answer.refusal_reason == "follow_up_needs_clarification"
    assert "şikâyetten vazgeçme" in answer.answer


def test_ambiguous_follow_up_fetches_nearby_sources_without_generation() -> None:
    """Belirsiz takip sorusu boş kalmaz; önceki kanun kapsamında yakın kaynak ister."""
    from RAG.agent.conversation import ConversationTurn, LegalConversation
    from RAG.agent.legal_agent import LegalAnswer

    class NearbyOnlyAgent:
        def __init__(self) -> None:
            self.query = ""

        def nearby_sources(self, query: str, **_: object) -> LegalAnswer:
            self.query = query
            return LegalAnswer(answer="Yakın kaynaklar:\n- TCK Madde 65 [S1]", grounded=False)

    agent = NearbyOnlyAgent()
    conversation = LegalConversation(agent=agent)  # type: ignore[arg-type]
    conversation.turns.append(
        ConversationTurn(
            question="Hırsız yakalandığında ne olur?",
            answer=LegalAnswer(answer="Önceki cevap.", grounded=True),
            primary_law_number="5237",
            primary_law_name="Türk Ceza Kanunu",
        )
    )

    result = conversation.ask("Peki, eğer onu affedersem ne olur?")

    assert result.related_to_previous is True
    assert agent.query.startswith("5237 sayılı Kanun kapsamında:")
    assert "Yakın kaynaklar" in result.answer.answer


def test_conversation_memory_is_sent_for_new_question_even_without_keyword_overlap() -> None:
    """Yeni soru ortak kelime taşımadığında da LLM önceki olay özetini görür."""
    from RAG.agent.conversation import ConversationTurn, LegalConversation
    from RAG.agent.legal_agent import LegalAnswer

    class MemoryAgent:
        def __init__(self) -> None:
            self.memory = ""

        def answer(self, _: str, **kwargs: object) -> LegalAnswer:
            self.memory = str(kwargs.get("conversation_memory") or "")
            return LegalAnswer(answer="Yeni cevap.", grounded=False)

    agent = MemoryAgent()
    conversation = LegalConversation(agent=agent)  # type: ignore[arg-type]
    conversation.turns.append(
        ConversationTurn(
            question="Arabam çalınırsa ne yapmalıyım?",
            answer=LegalAnswer(answer="Araç hırsızlığı hakkında önceki cevap.", grounded=True),
            primary_law_number="5237",
            primary_law_name="Türk Ceza Kanunu",
        )
    )

    result = conversation.ask("Hırsız yakalandığında başına ne gelecek?")

    assert result.related_to_previous is False
    assert result.memory_used is True
    assert "Arabam çalınırsa" in agent.memory


def test_model_abstention_cannot_be_marked_as_grounded() -> None:
    """Modelin kanıt yetersizliği beyanına otomatik kaynak etiketi eklenmez."""
    from RAG.agent.legal_agent import LegalRagAgent

    assert LegalRagAgent._model_abstained("Sağlanan kaynaklarda doğrulanamadı.") is True
    assert LegalRagAgent._model_abstained("Madde 81 müebbet hapis öngörür. [S1]") is False


def test_repeated_long_sentences_are_removed_before_caching() -> None:
    """Tekrarlayan üretim kullanıcıya ve kalıcı cache'e aktarılmaz."""
    from RAG.agent.legal_agent import LegalRagAgent

    repeated = (
        "Yakalanan kişinin cezası olayın özelliklerine göre belirlenir. "
        "Yakalanan kişinin cezası olayın özelliklerine göre belirlenir."
    )

    assert LegalRagAgent._has_excessive_repetition(repeated) is True
    assert LegalRagAgent._remove_repeated_sentences(repeated).count("özelliklerine göre") == 1


def test_abstention_shows_nearby_laws_without_claiming_a_direct_answer() -> None:
    """Eksik kanıtta yakın kaynaklar bilgi amaçlı gösterilir; cevap doğrulanmış sayılmaz."""
    from RAG.agent.legal_agent import LegalRagAgent

    answer = LegalRagAgent._nearby_evidence_refusal(
        [{"label": "S1", "law_name": "Vergi Usul Kanunu", "article_number": "5"}],
        total_ms=20,
        retrieval_ms=5,
        generation_ms=10,
        plan_name="semantic_fast",
        plan_reason="Test",
        draft_answer="Sağlanan kaynaklarda doğrulanamadı.",
    )

    assert answer.grounded is False
    assert "doğrudan cevabını" in answer.answer
    assert "Vergi Usul Kanunu" in answer.answer
    assert "kesin cevabı" in answer.answer


def test_query_transform_corrects_typos_without_adding_legal_facts() -> None:
    """Dönüşüm, özgün soruyu korur ve yalnız güvenli yazım düzeltmesi ekler."""
    from RAG.retriever.query_transform import _deterministic_queries

    variants = _deterministic_queries("vergi nede sorgulabilir")

    assert variants[0] == "vergi nede sorgulabilir"
    assert "vergi nerede sorgulanabilir" in variants
    assert all("213" not in item for item in variants)


def test_llm_query_variants_are_recorded_separately() -> None:
    """Arayüz, LLM'in ürettiği sorguları kural tabanlı düzeltmeden ayırabilir."""
    from RAG.retriever.query_transform import QueryTransform

    transformed = QueryTransform(
        original="hırsız yakalanınca ne olur",
        queries=["hırsız yakalanınca ne olur", "hırsızlık suçunda yakalama sonrası süreç"],
        used_llm=True,
        llm_queries=["hırsızlık suçunda yakalama sonrası süreç"],
    )

    assert transformed.used_llm is True
    assert transformed.llm_queries == ["hırsızlık suçunda yakalama sonrası süreç"]


def test_query_transform_rejects_shortened_llm_query() -> None:
    """LLM, özgün sorudan daha az bilgi taşıyan kısaltmayı aramaya sokamaz."""
    from RAG.retriever.query_transform import _is_useful_llm_variant

    assert _is_useful_llm_variant("vergi nede sorgulabilir", "vergi nede") is False
    assert _is_useful_llm_variant("vergi nede sorgulabilir", "vergi nerede sorgulanabilir") is True
