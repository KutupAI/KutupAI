"""Çok turlu hukukî sohbet için hafif oturum belleği.

Bu katman önceki cevabı yeniden retrieval'a göndermek yerine, takip sorusunu
son güvenilir kanun bağlamına daraltır. Böylece yeni bilgi aranır; eski kanıt
ve cevap gereksiz yere yeniden üretilmez.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from RAG.agent.legal_agent import LegalAnswer, LegalRagAgent
from RAG.retriever.text_utils import fold_turkish, tokenize


_FOLLOW_UP_MARKERS = (
    "peki", "ayrica", "ayrıca", "bunun", "buna", "bunu", "bu durumda",
    "devam", "devami", "devamı", "ayni", "aynı", "onun", "orada",
    "ikinci", "sonra", "daha", "ne kadar", "hangi sure", "hangi süre",
)
_GENERIC_TERMS = {"kanun", "madde", "nedir", "nasil", "nasıl", "gore", "göre", "ile", "icin", "için"}
_AMBIGUOUS_FOLLOW_UP = (
    "onu affedersem", "bunu affedersem", "affedersem", "bu durumda yapmam gereken",
    "bu durumda ne yap", "peki ne yapmam gerek", "peki ne yapmaliyim",
)


@dataclass(frozen=True)
class ConversationTurn:
    """Oturumda saklanan, yeniden retrieval gerektirmeyen önceki tur özeti."""

    question: str
    answer: LegalAnswer
    primary_law_number: Optional[str]
    primary_law_name: Optional[str]


@dataclass(frozen=True)
class ConversationResult:
    """Kullanıcıya gösterilecek cevap ile konuşma-bağlantı bilgisini taşır."""

    answer: LegalAnswer
    related_to_previous: bool
    memory_used: bool
    retrieval_query: str
    previous_turn: Optional[ConversationTurn]


class LegalConversation:
    """Tek kullanıcılı, bellek içi hukukî sohbet oturumu."""

    def __init__(self, agent: Optional[LegalRagAgent] = None) -> None:
        self.agent = agent or LegalRagAgent()
        self.turns: List[ConversationTurn] = []

    @staticmethod
    def _primary_source(answer: LegalAnswer) -> tuple[Optional[str], Optional[str]]:
        """İlk güvenilir kaynaktan sonraki soru için kanun kapsamı çıkarır."""
        if not answer.grounded or not answer.sources:
            return None, None
        source = answer.sources[0]
        law = str(source.get("law_number") or "").strip()
        return (law if law and law != "unknown" else None), str(source.get("law_name") or "").strip() or None

    @staticmethod
    def _has_explicit_new_law(question: str) -> bool:
        """Yeni bir kanun numarası veya adı varsa önceki konuya zorla bağlamaz."""
        normalized = fold_turkish(question).casefold()
        return "sayili" in normalized or any(name in normalized for name in ("kanunu", "kanununda", "cmk", "tck", "kvkk"))

    def _is_related(self, question: str, previous: ConversationTurn) -> bool:
        """Takip işaretleri ve anlamlı kelime kesişimiyle konu devamını saptar."""
        normalized = fold_turkish(question).casefold()
        if self._has_explicit_new_law(question):
            return False
        if any(marker in normalized for marker in _FOLLOW_UP_MARKERS):
            return True
        current_terms = set(tokenize(normalized, min_len=4)) - _GENERIC_TERMS
        previous_terms = set(tokenize(fold_turkish(previous.question).casefold(), min_len=4)) - _GENERIC_TERMS
        return bool(current_terms & previous_terms)

    def _build_memory(self, *, max_turns: int = 3, answer_chars: int = 700) -> str:
        """Son turları sınırlı bir özet olarak LLM'e taşır; retrieval metnine eklemez."""
        rows: List[str] = []
        for index, turn in enumerate(self.turns[-max_turns:], start=1):
            law = turn.primary_law_name or turn.primary_law_number or "Kanun bilgisi yok"
            answer = " ".join(turn.answer.answer.split())[:answer_chars]
            rows.append(f"Tur {index} | Konu: {turn.question}\nKanun: {law}\nÖnceki cevap özeti: {answer}")
        return "\n\n".join(rows)

    @staticmethod
    def _clarification_answer(question: str) -> Optional[LegalAnswer]:
        """Belirsiz kişi/işlem içeren takip sorularında yanlış hukukî varsayımı engeller."""
        normalized = fold_turkish(question).casefold()
        if not any(marker in normalized for marker in _AMBIGUOUS_FOLLOW_UP):
            return None
        if "affeder" in normalized:
            message = (
                "Buradaki “affetmek” ifadesi hukukta genel af, şikâyetten vazgeçme "
                "veya uzlaşma gibi farklı sonuçları olan işlemler anlamına gelebilir. "
                "Kesin yanıt için hangisini kastettiğinizi ve dosyanın soruşturma mı "
                "kovuşturma mı aşamasında olduğunu belirtin."
            )
        else:
            message = (
                "Bu takip sorusunda olayın hangi aşamada olduğu ve sizin rolünüz "
                "(mağdur, şikâyetçi veya başka bir kişi) belirtilmediği için güvenli bir "
                "işlem önerisi veremem. Olayı ve hedeflediğiniz işlemi kısaca netleştirin."
            )
        return LegalAnswer(answer=message, grounded=False, refusal_reason="follow_up_needs_clarification")

    def ask(self, question: str, *, top_k: int = 5, use_cache: bool = True) -> ConversationResult:
        """Soruyu yanıtlar; takip sorusunda yalnız yeni bilgiyi aynı kanunda arar."""
        clean_question = " ".join((question or "").split())
        previous = self.turns[-1] if self.turns else None
        conversation_memory = self._build_memory()
        related = bool(previous and self._is_related(clean_question, previous))
        retrieval_query = clean_question

        # Önceki kanun kesin ise yeni soruya yalnız bu kapsam eklenir. Önceki
        # soru/cevap arama metnine eklenmez; bu nedenle eski bilgi tekrar aranmaz.
        if related and previous and previous.primary_law_number:
            retrieval_query = f"{previous.primary_law_number} sayılı Kanun kapsamında: {clean_question}"

        # Kişisel zamirlerle kurulan belirsiz takiplerde önceki kanunu otomatik
        # varsaymak (ör. “affetmek”i genel af saymak) ciddi hukukî hata yaratır.
        # Yine de kullanıcı boş yanıt almaz: aynı kanunda yalnız yakın kaynaklar
        # aranır; LLM'e doğrulanmamış sonuç ürettirilmez.
        clarification = self._clarification_answer(clean_question) if related else None
        if clarification:
            answer = self.agent.nearby_sources(
                retrieval_query,
                top_k=top_k,
                conversation_memory=conversation_memory,
                message=(
                    f"{clarification.answer}\n\nSağlanan kaynaklarda bu takip sorusunun "
                    "doğrudan cevabı bulunamadı. Aşağıdaki maddeler önceki konuyla genel "
                    "olarak ilgili olabilir; kesin cevap olarak değerlendirilmemelidir."
                ),
            )
            self.turns.append(
                ConversationTurn(
                    question=clean_question,
                    answer=answer,
                    primary_law_number=None,
                    primary_law_name=None,
                )
            )
            return ConversationResult(
                answer=answer,
                related_to_previous=True,
                memory_used=bool(conversation_memory),
                retrieval_query=retrieval_query,
                previous_turn=previous,
            )

        answer = self.agent.answer(
            retrieval_query,
            top_k=top_k,
            use_cache=use_cache,
            conversation_memory=conversation_memory or None,
        )
        law_number, law_name = self._primary_source(answer)
        self.turns.append(
            ConversationTurn(
                question=clean_question,
                answer=answer,
                primary_law_number=law_number,
                primary_law_name=law_name,
            )
        )
        return ConversationResult(
            answer=answer,
            related_to_previous=related,
            memory_used=bool(conversation_memory),
            retrieval_query=retrieval_query,
            previous_turn=previous if related else None,
        )

    def clear(self) -> None:
        """Yalnız oturum belleğini temizler; disk üzerindeki semantic cache korunur."""
        self.turns.clear()
