from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from dataclasses import asdict, dataclass, field
import re
from time import perf_counter
from typing import Dict, List, Optional

from Inference.client.inference_request import InferenceRequest, Message
from Inference.client.llama_client import LlamaClient

from RAG.agent.citations import (
    render_citations,
    validate_citations,
)

from RAG.agent.context_builder import (
    _article_scoped_text,
    build_context,
)

from RAG.agent.semantic_cache import SemanticCache

from RAG.retriever.query_intent import service_lookup_notice
from RAG.retriever.retriever import retrieve
from RAG.retriever.text_utils import fold_turkish, tokenize

from RAG.configuration.rag_config_loader import (
    agent_config,
    cache_config,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

_SYSTEM_PROMPT = """
Sen, yalnızca sana verilen HUKUKİ BAĞLAM içindeki kaynakları kullanarak
cevap veren dikkatli bir Türk hukuku RAG asistanısın.

Görevin sadece kaynaklardan cümle kopyalamak değildir.
Önce kullanıcının sorusunu dikkatlice anlamalı, ardından verilen bütün
kaynak parçalarını birlikte incelemeli ve soruya kaynakların desteklediği
en doğru ve en faydalı cevabı vermelisin.

CEVAP ÜRETME SIRASI:

1. Önce soruyu yeniden değerlendir:
   - Kullanıcı tam olarak ne soruyor?
   - Belirli bir kanun, madde, kurum, süre, şart veya hukuki sonuç mu soruyor?
   - Soru yalnızca bir madde numarası içeriyorsa hangi kanunun kastedildiğini
     bağlam destekliyor mu?
   - Soruyu yalnız ilk retrieval sonucuna bakarak yorumlama.

2. Ardından HUKUKİ BAĞLAM içindeki TÜM kaynak parçalarını incele:
   - Sorunun doğrudan cevabını içeren pasaj var mı?
   - Cevap birden fazla chunk'a dağılmış olabilir mi?
   - Aynı kanunun aynı maddesine ait parçalar birbirini tamamlıyor mu?
   - Yakın fakat farklı bir madde doğrudan cevapmış gibi görünüyor mu?

3. Doğrudan cevap kaynaklarda bulunuyorsa:
   - Kaynaklardaki bilgileri birleştirerek açık, doğal ve kısa bir cevap ver.
   - Kaynakta açıkça bulunmayan hiçbir şart, istisna, süre, ceza,
     kurum, sonuç veya yorum ekleme.
   - Gerekirse birden fazla chunk'taki bilgiyi birleştir.
   - Kullanıcının sorduğu noktaya doğrudan cevap ver.
   - Her somut hukukî iddiayı onu gerçekten destekleyen [S1], [S2]
     gibi kaynak etiketiyle destekle.

4. Kaynaklar yalnızca cevabın bir bölümünü destekliyorsa:
   - Desteklenen kısmı açıkça söyle.
   - Desteklenmeyen kısmı tahmin etme.
   - Şunu açıkça belirt:
     "Sorunun bu kısmına ilişkin doğrudan bir hüküm sağlanan kaynaklarda doğrulanamadı."

5. Sorunun doğrudan cevabı kaynaklarda bulunmuyorsa:
   - Kesinlikle tahmin yürütme.
   - İlk olarak açıkça şunu söyle:
     "Sağlanan kaynaklarda sorunun doğrudan cevabı doğrulanamadı."
   - Eğer bağlamda konuya gerçekten yakın kanun veya maddeler varsa,
     "Yakın kaynaklar:" başlığı altında bunları kısa şekilde açıkla.
   - Yakın kaynakların sorunun kesin cevabı olmadığını mutlaka belirt.
   - Yakın kaynaklardan yeni bir hukuki sonuç çıkarma.
   - Sadece kaynaklarda açıkça yazan bilgiyi aktar.

6. Kaynaklar konu dışıysa:
   - Sadece:
     "Sağlanan kaynaklarda sorunun doğrudan cevabı doğrulanamadı."
     de.
   - İlgisiz maddeleri sırf sonuç üretmek için kullanma.

7. Açık kanun veya madde referanslarında:
   - Kullanıcının belirttiği kanun/madde mevcutsa öncelikle o kaynağı değerlendir.
   - Farklı bir kanunun aynı numaralı maddesini otomatik olarak kullanma.
   - Kullanıcı yalnız "5. madde" gibi bir ifade kullanmış ve hangi kanunun
     kastedildiği kaynaklardan güvenilir şekilde belirlenemiyorsa bunu belirt.
   - Semantic benzerlik tek başına kanun kimliği varsaymak için yeterli değildir.

8. Kaynak kullanımı:
   - Önceki konuşma cevaplarını hukukî kaynak olarak kullanma.
   - Genel dünya bilgini veya ezberindeki Türk hukukunu kullanma.
   - İnternetten öğrenilmiş olabilecek bilgileri kullanma.
   - Yalnızca güncel HUKUKİ BAĞLAM kullanılabilir.

9. Kaynak etiketleri:
   - Yalnızca bağlamda gerçekten bulunan [S1], [S2], [S3] etiketlerini kullan.
   - Kaynak etiketi uydurma.
   - Bir iddiayı desteklemeyen kaynağı o iddiaya ekleme.
   - Aynı paragraftaki birkaç cümle aynı kaynaktan destekleniyorsa
     paragraf sonunda tek kaynak etiketi kullanılabilir.

10. Üslup:
   - Türkçe cevap ver.
   - Önce doğrudan cevabı söyle.
   - Gereksiz uzun giriş yapma.
   - Bilgilendirici, açık ve ihtiyatlı ol.

EN ÖNEMLİ KURAL:

Kaynaklarda olmayan bir bilgiyi doğru olabileceğini düşünsen bile ASLA ekleme.
Kaynaklar yeterli değilse daha az cevap vermek, uydurma cevap vermekten daha iyidir.
"""


# ============================================================
# CONVERSATION MEMORY RULE
# ============================================================

_CONVERSATION_MEMORY_RULE = """

KONUŞMA BELLEĞİ KURALI:

ÖNCEKİ KONUŞMA BELLEĞİ yalnızca:
- zamirleri,
- kişileri,
- olayları,
- daha önce açıkça belirtilen kanun veya konu referanslarını

anlamak için kullanılabilir.

Önceki asistan cevapları bağımsız hukukî kanıt değildir.

Yeni bir hukukî iddia yalnızca güncel HUKUKİ BAĞLAM içindeki
kaynaklardan desteklenebilir.

Kullanıcı açıkça yeni bir konu açmışsa eski konuşma bağlamını cevaba taşıma.
"""


# ============================================================
# NEARBY SOURCES PROMPT
# ============================================================

_NEARBY_SOURCES_PROMPT = """
Sen hukukî kaynakları dikkatli şekilde açıklayan bir Türkçe RAG asistanısın.

Bu aşamaya gelinmesinin nedeni, kullanıcının sorusunun DOĞRUDAN cevabının
sağlanan kaynaklarda güvenilir şekilde doğrulanamamış olmasıdır.

Görevin cevap uydurmak değildir.

Kurallar:

1. İlk cümlede:
   "Sağlanan kaynaklarda sorunun doğrudan cevabı doğrulanamadı."
   de.

2. Kaynaklar arasında soruya gerçekten yakın içerik varsa:
   "Yakın kaynaklar:" başlığı aç.

3. Her yakın kaynak için yalnızca o kaynakta açıkça yazan bilgiyi
   kısa şekilde özetle.

4. Her açıklamanın sonunda doğru [S1], [S2] gibi etiketi kullan.

5. Yakın kaynağın neden ilgili olduğunu açıklayabilirsin ancak
   bundan yeni hukukî sonuç çıkaramazsın.

6. Şunları yapma:
   - Eksik cevabı genel hukuk bilgisiyle tamamlama.
   - Yakın bir maddeyi doğrudan cevap olarak gösterme.
   - Kaynakta olmayan kurum, süre, şart, ceza veya prosedür ekleme.

7. Hiçbir kaynak gerçekten ilgili değilse yalnızca:
   "Sağlanan kaynaklarda sorunun doğrudan cevabı doğrulanamadı."
   de.
"""


# ============================================================
# CITATION REPAIR PROMPT
# ============================================================

_CITATION_REPAIR_PROMPT = """
Sen hukukî bir cevabın kaynak doğrulamasını yapan editörsün.

Görevin yeni bilgi üretmek değildir.

Kurallar:

1. Cevaptaki her hukukî iddiayı HUKUKİ BAĞLAM ile karşılaştır.
2. Bağlam tarafından desteklenmeyen iddiaları sil.
3. Desteklenen iddialara doğru [S1], [S2] gibi kaynak etiketlerini ekle.
4. Kaynak etiketi uydurma.
5. Cevabın doğrudan kısmı kaynaklarda yoksa tahmin yürütme.
6. Böyle bir durumda:
   "Sağlanan kaynaklarda sorunun doğrudan cevabı doğrulanamadı."
   ifadesini kullan.
7. Yalnız kaynaklarda bulunan bilgileri kullan.
8. Yeni hukuk bilgisi ekleme.
9. Türkçe ve kısa cevap ver.
"""


# ============================================================
# REGEX
# ============================================================

_MODEL_ABSTENTION = re.compile(
    r"\b("
    r"saglanan\s+kaynaklarda\s+(?:bu\s+soruyu\s+)?dogrulanamadi|"
    r"dogrulanamiyor|"
    r"yeterli\s+(?:hukuki\s+)?(?:bilgi|pasaj|kaynak)\s+(?:bulunamadi|yok)|"
    r"guvenilir\s+bir\s+(?:bilgi|yonlendirme)\s+veremem"
    r")\b",
    re.IGNORECASE,
)

_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+|\n+"
)


# ============================================================
# RESULT MODEL
# ============================================================

@dataclass(frozen=True)
class LegalAnswer:

    answer: str

    sources: List[Dict[str, object]] = field(
        default_factory=list
    )

    citations: str = ""

    grounded: bool = False

    cache_hit: bool = False

    cache_similarity: Optional[float] = None

    retrieval_ms: float = 0.0

    generation_ms: float = 0.0

    total_ms: float = 0.0

    prompt_tokens: int = 0

    completion_tokens: int = 0

    tokens_per_second: Optional[float] = None

    ttft_ms: Optional[float] = None

    refusal_reason: Optional[str] = None

    retrieval_plan: Optional[str] = None

    retrieval_plan_reason: Optional[str] = None

    draft_answer: Optional[str] = None

    def to_dict(
        self
    ) -> Dict[str, object]:

        return asdict(
            self
        )

    @classmethod
    def from_dict(
        cls,
        value: Dict[str, object],
        **overrides: object,
    ) -> "LegalAnswer":

        allowed = {
            name: value.get(name)
            for name in cls.__dataclass_fields__
            if name in value
        }

        allowed.update(
            overrides
        )

        return cls(
            **allowed
        )


# ============================================================
# AGENT
# ============================================================

class LegalRagAgent:

    def __init__(
        self,
        *,
        base_url: str = agent_config.base_url,
        timeout_seconds: int = agent_config.timeout_seconds,
        cache: Optional[SemanticCache] = None,
    ) -> None:

        self.client = LlamaClient(
            base_url=base_url,
            timeout=timeout_seconds,
        )

        self.cache = (
            cache
            or SemanticCache()
        )

    # ========================================================
    # REFUSAL
    # ========================================================

    @staticmethod
    def _refusal(
        reason: str,
        total_ms: float = 0.0,
        draft_answer: Optional[str] = None,
    ) -> LegalAnswer:

        return LegalAnswer(
            answer=(
                "Sağlanan kaynaklarda sorunun doğrudan cevabı "
                "doğrulanamadı."
            ),
            grounded=False,
            total_ms=round(
                total_ms,
                3,
            ),
            refusal_reason=reason,
            draft_answer=draft_answer,
        )

    # ========================================================
    # ABSTENTION CHECK
    # ========================================================

    @staticmethod
    def _model_abstained(
        text: str,
    ) -> bool:

        normalized = fold_turkish(
            text or ""
        ).casefold()

        return bool(
            _MODEL_ABSTENTION.search(
                normalized
            )
        )

    # ========================================================
    # REMOVE REPETITION
    # ========================================================

    @staticmethod
    def _remove_repeated_sentences(
        text: str,
    ) -> str:

        sentences = _SENTENCE_BOUNDARY.split(
            (text or "").strip()
        )

        seen: set[str] = set()

        cleaned: List[str] = []

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            key = " ".join(
                fold_turkish(
                    sentence
                )
                .casefold()
                .split()
            )

            if (
                len(key) >= 50
                and key in seen
            ):
                continue

            if len(key) >= 50:
                seen.add(
                    key
                )

            cleaned.append(
                sentence
            )

        return "\n\n".join(
            cleaned
        )

    @classmethod
    def _has_excessive_repetition(
        cls,
        text: str,
    ) -> bool:

        return (
            cls._remove_repeated_sentences(
                text
            )
            != (text or "").strip()
        )

    # ========================================================
    # NEARBY REFUSAL
    # ========================================================

    @staticmethod
    def _nearby_evidence_refusal(
        sources: List[Dict[str, object]],
        *,
        total_ms: float,
        retrieval_ms: float,
        generation_ms: float,
        plan_name: str,
        plan_reason: str,
        draft_answer: str,
    ) -> LegalAnswer:

        nearby: List[str] = []

        for source in sources[:3]:

            law_number = str(
                source.get(
                    "law_number"
                )
                or ""
            ).strip()

            law_name = str(
                source.get(
                    "law_name"
                )
                or law_number
                or "Bilinmeyen kanun"
            ).strip()

            article = str(
                source.get(
                    "article_number"
                )
                or source.get(
                    "article_no"
                )
                or "-"
            )

            label = str(
                source.get(
                    "label"
                )
                or ""
            )

            if (
                law_number
                and law_number != "unknown"
            ):

                clean_law_name = re.sub(
                    rf"^{re.escape(law_number)}[_\s-]*",
                    "",
                    law_name,
                ).strip()

                law_name = (
                    f"{law_number} sayılı "
                    f"{clean_law_name}"
                )

            nearby.append(
                f"- {law_name}, "
                f"Madde {article} [{label}]"
            )

        answer = (
            "Sağlanan kaynaklarda sorunun doğrudan cevabı "
            "doğrulanamadı."
        )

        if nearby:

            answer += (
                "\n\nYakın kaynaklar:\n"
                + "\n".join(
                    nearby
                )
                + "\n\nBu kaynaklar konuya yakın olmakla birlikte "
                "sorunun kesin cevabı olarak değerlendirilmemelidir."
            )

        return LegalAnswer(
            answer=answer,
            sources=sources,
            citations=render_citations(
                sources
            ),
            grounded=False,
            retrieval_ms=round(
                retrieval_ms,
                3,
            ),
            generation_ms=round(
                generation_ms,
                3,
            ),
            total_ms=round(
                total_ms,
                3,
            ),
            refusal_reason=(
                "direct_answer_not_found_nearby_sources_shown"
            ),
            retrieval_plan=plan_name,
            retrieval_plan_reason=plan_reason,
            draft_answer=draft_answer,
        )

    # ========================================================
    # ARTICLE LOOKUP
    # ========================================================

    @staticmethod
    def _best_article_lookup_source(
        question: str,
        sources: List[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:

        ignored = {
            "kanun",
            "kanununda",
            "kanunu",
            "madde",
            "maddede",
            "hangi",
            "nedir",
            "duzenlenmistir",
            "duzenlenir",
            "gore",
            "sayili",
        }

        terms = [
            term
            for term in tokenize(
                fold_turkish(
                    question
                ),
                min_len=4,
            )
            if term not in ignored
        ]

        if not terms:
            return None

        best: Optional[
            Dict[str, object]
        ] = None

        best_score = 0

        for source in sources:

            haystack = fold_turkish(
                str(
                    source.get(
                        "text"
                    )
                    or ""
                )
            ).casefold()

            score = sum(
                term[:6] in haystack
                for term in terms
            )

            article = re.escape(
                str(
                    source.get(
                        "article_number"
                    )
                    or ""
                )
            )

            if (
                article
                and re.match(
                    rf"\s*madde\s+{article}\s*[–—-]",
                    haystack,
                )
            ):

                score += 10

            if score > best_score:

                best = source

                best_score = score

        return (
            best
            if best_score
            else None
        )

    # ========================================================
    # CITATION REPAIR
    # ========================================================

    def _repair_answer_citations(
        self,
        *,
        question: str,
        context_text: str,
        draft_answer: str,
    ):

        prompt = (
            f"SORU:\n"
            f"{question}\n\n"

            f"HUKUKİ BAĞLAM:\n"
            f"{context_text}\n\n"

            f"KONTROL EDİLECEK CEVAP:\n"
            f"{draft_answer}\n\n"

            f"DÜZELTİLMİŞ CEVAP:"
        )

        return self.client.generate(
            InferenceRequest(
                messages=[
                    Message(
                        role="system",
                        content=_CITATION_REPAIR_PROMPT,
                    ),
                    Message(
                        role="user",
                        content=prompt,
                    ),
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=agent_config.max_tokens,
            )
        )

    # ========================================================
    # ANSWER
    # ========================================================

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        use_cache: bool = True,
        conversation_memory: Optional[str] = None,
    ) -> LegalAnswer:

        started = perf_counter()

        question = " ".join(
            (question or "").split()
        )

        if not question:

            return self._refusal(
                "empty_question"
            )

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        if (
            use_cache
            and cache_config.enabled
            and not conversation_memory
        ):

            hit = self.cache.get(
                question
            )

            if hit:

                cached = LegalAnswer.from_dict(
                    hit.payload,
                    cache_hit=True,
                    cache_similarity=hit.similarity,
                    total_ms=round(
                        (
                            perf_counter()
                            - started
                        ) * 1000,
                        3,
                    ),
                )

                if not self._has_excessive_repetition(
                    cached.answer
                ):

                    return cached

        # ----------------------------------------------------
        # QUERY ROUTER
        # ----------------------------------------------------

        from RAG.retriever.query_router import (
            choose_query_plan,
        )

        plan = choose_query_plan(
            question
        )

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        retrieval_started = (
            perf_counter()
        )

        results = retrieve(
            question,
            top_k=top_k,
            mode=plan.mode,
            use_prf=plan.use_prf,
            use_reranker=plan.use_reranker,
            use_graph=plan.use_graph,
        )

        retrieval_ms = (
            perf_counter()
            - retrieval_started
        ) * 1000

        # ----------------------------------------------------
        # SERVICE LOOKUP
        # ----------------------------------------------------

        notice = service_lookup_notice(
            question,
            results,
        )

        if notice:

            return LegalAnswer(
                answer=(
                    "Bu soru bir hizmet veya portal işlemiyle ilgilidir. "
                    "Mevcut kaynaklarda işlemi doğrulayacak resmî kurum "
                    "rehberi bulunmadığından güvenilir bir işlem "
                    "yönlendirmesi veremem."
                ),
                grounded=False,
                retrieval_ms=round(
                    retrieval_ms,
                    3,
                ),
                total_ms=round(
                    (
                        perf_counter()
                        - started
                    ) * 1000,
                    3,
                ),
                refusal_reason=(
                    "official_service_source_missing"
                ),
                retrieval_plan=plan.name,
                retrieval_plan_reason=plan.rationale,
            )

        # ----------------------------------------------------
        # CONTEXT BUILD
        # ----------------------------------------------------

        context = build_context(
            results
        )

        if not context.sources:

            return self._refusal(
                "no_retrieved_evidence",
                (
                    perf_counter()
                    - started
                ) * 1000,
            )

        # ----------------------------------------------------
        # ARTICLE LOOKUP PREPARATION
        # ----------------------------------------------------

        result_text_by_id = {
            str(item["id"]):
                _article_scoped_text(
                    item
                )
            for item in results
        }

        internal_sources = [
            {
                **source,
                "text": result_text_by_id.get(
                    str(
                        source.get(
                            "chunk_id"
                        )
                    ),
                    "",
                ),
            }
            for source in context.sources
        ]

        # ----------------------------------------------------
        # ARTICLE LOOKUP
        # ----------------------------------------------------

        if plan.name == "article_lookup":

            selected = (
                self._best_article_lookup_source(
                    question,
                    internal_sources,
                )
            )

            if selected:

                answer = LegalAnswer(
                    answer=(
                        f"Soruda geçen konu, "
                        f"{selected['law_name']} kapsamındaki "
                        f"Madde {selected['article_number']}'te "
                        f"düzenlenmektedir. "
                        f"[{selected['label']}]"
                    ),
                    sources=context.sources,
                    citations=render_citations(
                        context.sources
                    ),
                    grounded=True,
                    retrieval_ms=round(
                        retrieval_ms,
                        3,
                    ),
                    total_ms=round(
                        (
                            perf_counter()
                            - started
                        ) * 1000,
                        3,
                    ),
                    retrieval_plan=plan.name,
                    retrieval_plan_reason=plan.rationale,
                )

                return answer

        # ----------------------------------------------------
        # MAIN PROMPT
        # ----------------------------------------------------

        prompt = (
            f"SORU:\n"
            f"{question}\n\n"

            f"ÖNCEKİ KONUŞMA BELLEĞİ:\n"
            f"{conversation_memory or '(yok)'}\n\n"

            f"HUKUKİ BAĞLAM:\n"
            f"{context.text}\n\n"

            f"GÖREV:\n"
            f"Soruyu yeniden değerlendir. "
            f"Yukarıdaki bütün kaynak parçalarını birlikte incele. "
            f"Sorunun doğrudan cevabı kaynaklarda varsa yalnız "
            f"bu kaynaklara dayanarak en iyi cevabı oluştur. "
            f"Doğrudan cevap yoksa bunu açıkça belirt ve yalnız "
            f"gerçekten ilgili yakın kaynakları ayrı olarak göster.\n\n"

            f"CEVAP:"
        )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        generation_started = (
            perf_counter()
        )

        response = self.client.generate(
            InferenceRequest(
                messages=[
                    Message(
                        role="system",
                        content=(
                            _SYSTEM_PROMPT
                            + _CONVERSATION_MEMORY_RULE
                        ),
                    ),
                    Message(
                        role="user",
                        content=prompt,
                    ),
                ],
                temperature=agent_config.temperature,
                top_p=agent_config.top_p,
                max_tokens=agent_config.max_tokens,
            )
        )

        generation_ms = (
            perf_counter()
            - generation_started
        ) * 1000

        if (
            not response.success
            or not response.text.strip()
        ):

            return self._refusal(
                "generation_unavailable",
                (
                    perf_counter()
                    - started
                ) * 1000,
            )

        response.text = (
            self._remove_repeated_sentences(
                response.text
            )
        )

        # ----------------------------------------------------
        # MODEL ABSTAINED
        # ----------------------------------------------------

        if self._model_abstained(
            response.text
        ):

            return self._nearby_evidence_refusal(
                context.sources,
                total_ms=(
                    perf_counter()
                    - started
                ) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                plan_name=plan.name,
                plan_reason=plan.rationale,
                draft_answer=response.text,
            )

        # ----------------------------------------------------
        # CITATION VALIDATION
        # ----------------------------------------------------

        grounded, invalid = (
            validate_citations(
                response.text,
                context.sources,
            )
        )

        if invalid:

            return self._nearby_evidence_refusal(
                context.sources,
                total_ms=(
                    perf_counter()
                    - started
                ) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                plan_name=plan.name,
                plan_reason=plan.rationale,
                draft_answer=response.text,
            )

        # ----------------------------------------------------
        # CITATION REPAIR
        # ----------------------------------------------------

        if not grounded:

            repair = self._repair_answer_citations(
                question=question,
                context_text=context.text,
                draft_answer=response.text,
            )

            if (
                not repair.success
                or not repair.text.strip()
            ):

                return self._nearby_evidence_refusal(
                    context.sources,
                    total_ms=(
                        perf_counter()
                        - started
                    ) * 1000,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    plan_name=plan.name,
                    plan_reason=plan.rationale,
                    draft_answer=response.text,
                )

            repaired_text = (
                self._remove_repeated_sentences(
                    repair.text
                )
            )

            repaired_grounded, repaired_invalid = (
                validate_citations(
                    repaired_text,
                    context.sources,
                )
            )

            if (
                repaired_invalid
                or not repaired_grounded
            ):

                return self._nearby_evidence_refusal(
                    context.sources,
                    total_ms=(
                        perf_counter()
                        - started
                    ) * 1000,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    plan_name=plan.name,
                    plan_reason=plan.rationale,
                    draft_answer=response.text,
                )

            response.text = repaired_text

            grounded = True

        # ----------------------------------------------------
        # FINAL METRICS
        # ----------------------------------------------------

        tokens_per_second = None

        if (
            response.completion_tokens
            and generation_ms > 0
        ):

            tokens_per_second = round(
                response.completion_tokens
                / (
                    generation_ms
                    / 1000
                ),
                3,
            )

        timings = (
            response.server_timings
            or {}
        )

        ttft_ms = timings.get(
            "prompt_ms"
        )

        answer = LegalAnswer(
            answer=response.text.strip(),
            sources=context.sources,
            citations=render_citations(
                context.sources
            ),
            grounded=True,
            retrieval_ms=round(
                retrieval_ms,
                3,
            ),
            generation_ms=round(
                generation_ms,
                3,
            ),
            total_ms=round(
                (
                    perf_counter()
                    - started
                ) * 1000,
                3,
            ),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            tokens_per_second=tokens_per_second,
            ttft_ms=(
                round(
                    float(
                        ttft_ms
                    ),
                    3,
                )
                if isinstance(
                    ttft_ms,
                    (
                        int,
                        float,
                    ),
                )
                else None
            ),
            retrieval_plan=plan.name,
            retrieval_plan_reason=plan.rationale,
        )

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        if (
            use_cache
            and cache_config.enabled
            and not conversation_memory
        ):

            self.cache.put(
                question,
                answer.to_dict(),
            )

        return answer


# ============================================================
# INTERACTIVE TEST WITH CONVERSATION MEMORY
# ============================================================

def main():

    print("=" * 70)
    print("KutupAI - Legal RAG Interactive Test")
    print("=" * 70)

    print(
        "\nKomutlar:\n"
        "  exit / quit / çık  -> Programdan çık\n"
        "  clear              -> Konuşma belleğini temizle\n"
    )

    print("Agent yükleniyor...")

    agent = LegalRagAgent()

    print("Agent hazır.")

    # --------------------------------------------------------
    # TEMPORARY CONVERSATION MEMORY
    # --------------------------------------------------------
    # Program kapandığında bu bellek tamamen silinir.
    #
    # Her eleman:
    # {
    #     "role": "user" | "assistant",
    #     "content": "..."
    # }
    # --------------------------------------------------------

    conversation_history = []

    # Son 6 mesajı tut:
    # 3 kullanıcı + 3 asistan mesajı gibi düşünülebilir.
    MAX_MEMORY_MESSAGES = 6

    def build_conversation_memory() -> str:
        """
        Son mesajlardan LLM'e gönderilecek konuşma belleğini oluşturur.

        Bu bellek yalnız mevcut sorudaki zamirleri, eksik ifadeleri
        ve önceki konu referanslarını anlamak içindir.

        Hukukî kaynak olarak kullanılmaz.
        """

        if not conversation_history:
            return ""

        recent_messages = conversation_history[-MAX_MEMORY_MESSAGES:]

        lines = []

        for message in recent_messages:

            if message["role"] == "user":
                role = "Kullanıcı"
            else:
                role = "Asistan"

            lines.append(
                f"{role}: {message['content']}"
            )

        return "\n".join(lines)

    while True:

        print("\n" + "=" * 70)

        question = input("\nQuestion: ").strip()

        if not question:
            continue

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if question.casefold() in {
            "exit",
            "quit",
            "çık",
            "cik",
        }:

            print("\nKutupAI kapatıldı.")
            break

        # ----------------------------------------------------
        # CLEAR MEMORY
        # ----------------------------------------------------

        if question.casefold() in {
            "clear",
            "temizle",
            "reset",
        }:

            conversation_history.clear()

            print("\nKonuşma belleği temizlendi.")

            continue

        # ----------------------------------------------------
        # BUILD MEMORY BEFORE ADDING CURRENT QUESTION
        # ----------------------------------------------------

        conversation_memory = build_conversation_memory()

        print("\nRAG çalışıyor...\n")

        try:

            result = agent.answer(
                question,
                top_k=5,
                use_cache=True,
                conversation_memory=(
                    conversation_memory
                    if conversation_memory
                    else None
                ),
            )

        except Exception as exc:

            print("\nERROR:")
            print(repr(exc))

            # Hatalı tur belleğe eklenmez.
            continue

        # ----------------------------------------------------
        # SAVE CURRENT TURN TO MEMORY
        # ----------------------------------------------------

        conversation_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        conversation_history.append(
            {
                "role": "assistant",
                "content": result.answer,
            }
        )

        # Belleğin kontrolsüz büyümesini engelle.
        if len(conversation_history) > MAX_MEMORY_MESSAGES:

            conversation_history = (
                conversation_history[-MAX_MEMORY_MESSAGES:]
            )

        # ====================================================
        # ANSWER
        # ====================================================

        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)

        print(result.answer)

        # ====================================================
        # SOURCES
        # ====================================================

        if result.sources:

            print("\n" + "=" * 70)
            print("SOURCES")
            print("=" * 70)

            for index, source in enumerate(
                result.sources,
                1,
            ):

                print(
                    f"\n[{index}] "
                    f"{source.get('law_name')}"
                )

                print(
                    f"Law No: "
                    f"{source.get('law_number')}"
                )

                print(
                    f"Article: "
                    f"{source.get('article_number')}"
                )

                print(
                    f"Label: "
                    f"{source.get('label')}"
                )

        # ====================================================
        # CITATIONS
        # ====================================================

        if result.citations:

            print("\n" + "=" * 70)
            print("CITATIONS")
            print("=" * 70)

            print(result.citations)

        # ====================================================
        # DEBUG
        # ====================================================

        print("\n" + "=" * 70)
        print("DEBUG")
        print("=" * 70)

        print(
            f"Grounded         : "
            f"{result.grounded}"
        )

        print(
            f"Cache Hit        : "
            f"{result.cache_hit}"
        )

        if result.cache_similarity is not None:

            print(
                f"Cache Similarity : "
                f"{result.cache_similarity:.4f}"
            )

        print(
            f"Retrieval Plan   : "
            f"{result.retrieval_plan}"
        )

        print(
            f"Plan Reason      : "
            f"{result.retrieval_plan_reason}"
        )

        print(
            f"Retrieval Time   : "
            f"{result.retrieval_ms:.2f} ms"
        )

        print(
            f"Generation Time  : "
            f"{result.generation_ms:.2f} ms"
        )

        print(
            f"Total Time       : "
            f"{result.total_ms:.2f} ms"
        )

        print(
            f"Prompt Tokens    : "
            f"{result.prompt_tokens}"
        )

        print(
            f"Completion Tokens: "
            f"{result.completion_tokens}"
        )

        if result.tokens_per_second is not None:

            print(
                f"Tokens / Second  : "
                f"{result.tokens_per_second:.2f}"
            )

        if result.ttft_ms is not None:

            print(
                f"TTFT             : "
                f"{result.ttft_ms:.2f} ms"
            )

        if result.refusal_reason:

            print(
                f"Refusal Reason   : "
                f"{result.refusal_reason}"
            )

        # ====================================================
        # MEMORY DEBUG
        # ====================================================

        print(
            f"Memory Messages  : "
            f"{len(conversation_history)}/"
            f"{MAX_MEMORY_MESSAGES}"
        )

        if result.draft_answer:

            print("\n--- Draft Answer ---")
            print(result.draft_answer)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()