"""Grounded Qwen legal-answering agent with context, citations, and cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from time import perf_counter
from typing import Dict, List, Optional

from Inference.client.inference_request import InferenceRequest, Message
from Inference.client.llama_client import LlamaClient
from RAG.agent.citations import render_citations, validate_citations
from RAG.agent.context_builder import _article_scoped_text, build_context
from RAG.agent.evidence_guards import (
    court_date_comparison_is_incomplete,
    direct_cross_law_reference_answer,
    direct_constitutional_annulment_answer,
    direct_travel_duration_answer,
    incomplete_court_date_comparison_answer,
)
from RAG.agent.semantic_cache import SemanticCache
from RAG.retriever.query_intent import service_lookup_notice
from RAG.retriever.retriever import retrieve
from RAG.retriever.text_utils import fold_turkish, tokenize
from RAG.configuration.rag_config_loader import agent_config, cache_config


_SYSTEM_PROMPT = """Sen yalnızca verilen hukuk bağlamına dayanarak cevap veren dikkatli bir Türk hukuk asistanısın.
Bağlamda bulunmayan bir bilgi, kanun, madde, tarih veya sonuç uydurma. Her somut hukuk
iddiasının sonuna, yalnızca bağlamdaki [S1], [S2] gibi bir kaynak etiketi koy. Bağlam
yetersizse açıkça 'Sağlanan kaynaklarda doğrulanamadı' de. Bir maddede yazmayan yeni bir
liste maddesi, istisna veya hukuki sonuç ekleme; özellikle komşu maddeleri (ör. özel
nitelikli veri şartları) sorulan maddeyle karıştırma. Kesin hukukî tavsiye verme;
bilgilendirici ve ihtiyatlı ol. Doğrudan cevap bağlamda yoksa bunu açıkça belirt;
varsa yalnız konuya yakın kanun/madde kaynaklarını “yakın kaynaklar” olarak listele ve
onların sorunun kesin cevabı olmadığını söyle. Cevabı Türkçe, kısa paragraflarla yaz."""

_REFERENCE_DOCUMENT_RULE = """\nREFERANS BELGE KURALI:
Bağlamda “Referans Belge” yazıyorsa bu kaynak kanun veya resmî hukukî hüküm değildir.
Yalnız belgenin yapısı, alanları veya örnek içeriği hakkında konuş; ondan hukukî sonuç,
zorunluluk veya güncel işlem kuralı çıkarma."""

_CONVERSATION_MEMORY_RULE = """\n\nKONUŞMA BELLEĞİ KURALI:
Aşağıdaki önceki konuşma yalnız soru içindeki kişi, olay ve zamirleri anlamak içindir.
Önceki cevapları bağımsız hukukî kaynak gibi kullanma; yeni hukukî iddiaları yalnız güncel
HUKUKİ BAĞLAM'daki kaynaklarla destekle. Kullanıcı açıkça yeni bir konu açarsa eski belleği
cevaba taşımama."""

_MODEL_ABSTENTION = re.compile(
    r"\b(saglanan\s+kaynaklarda\s+(?:bu\s+soruyu\s+)?dogrulanamadi|"
    r"dogrulanamiyor|yeterli\s+(?:hukuki\s+)?(?:bilgi|pasaj|kaynak)\s+(?:bulunamadi|yok)|"
    r"guvenilir\s+bir\s+(?:bilgi|yonlendirme)\s+veremem)\b",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

_NEARBY_SOURCES_PROMPT = """Sen hukukî kaynakları düzenleyen dikkatli bir Türkçe editörsün.
Sorunun doğrudan cevabı verilen bağlamda doğrulanmış değildir. Görevin yeni bir hukukî
sonuç, tavsiye, ceza, süre veya işlem uydurmak değil; yalnızca bağlamdaki en yakın
maddeleri kullanıcının anlayacağı şekilde kısa ve düzenli biçimde sunmaktır.

Şu kurallara uy:
1. İlk cümlede doğrudan cevabın kaynaklarda bulunamadığını veya doğrulanamadığını söyle.
2. Ardından yalnız bağlamda açıkça geçen bilgileri “Yakın kaynaklar” başlığı altında özetle.
3. Her madde/iddia için “5237 sayılı Türk Ceza Kanunu, Madde 65 [S1]” biçimini kullan:
   kanun adı ve madde numarası Türkçe görünür olmalı; [S1] etiketi yalnız doğrulama içindir.
4. Yakın kaynaklardan kesin bir sonuç çıkarma; eksik bilgi varsa bunu açıkça bırak.
5. Bağlamda yazmayan hiçbir bilgi ekleme."""


@dataclass(frozen=True)
class LegalAnswer:
    answer: str
    sources: List[Dict[str, object]] = field(default_factory=list)
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
    # Yalnız benchmark tanısı için saklanır; normal komut satırında gösterilmez.
    draft_answer: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, object], **overrides: object) -> "LegalAnswer":
        allowed = {name: value.get(name) for name in cls.__dataclass_fields__ if name in value}
        allowed.update(overrides)
        return cls(**allowed)


class LegalRagAgent:
    def __init__(
        self,
        *,
        base_url: str = agent_config.base_url,
        timeout_seconds: int = agent_config.timeout_seconds,
        cache: Optional[SemanticCache] = None,
    ) -> None:
        self.client = LlamaClient(base_url=base_url, timeout=timeout_seconds)
        self.cache = cache or SemanticCache()

    @staticmethod
    def _refusal(reason: str, total_ms: float = 0.0, draft_answer: Optional[str] = None) -> LegalAnswer:
        return LegalAnswer(
            answer="Sağlanan kaynaklarda bu soruyu doğrulayacak yeterli hukukî pasaj bulunamadı.",
            grounded=False,
            total_ms=round(total_ms, 3),
            refusal_reason=reason,
            draft_answer=draft_answer,
        )

    @staticmethod
    def _model_abstained(text: str) -> bool:
        """Modelin açıkça kanıt yetersizliği bildirdiği yanıtı kaynaklı saymaz."""
        return bool(_MODEL_ABSTENTION.search(fold_turkish(text or "").casefold()))

    @staticmethod
    def _remove_repeated_sentences(text: str) -> str:
        """Quantize modelin aynı uzun cümleyi art arda üretmesini temizler."""
        sentences = _SENTENCE_BOUNDARY.split((text or "").strip())
        seen: set[str] = set()
        cleaned: List[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            key = " ".join(fold_turkish(sentence).casefold().split())
            # Kısa kaynak etiketleri ve madde başlıkları tekrar edebilir; yalnız
            # anlam taşıyan, uzun cümlelerin birebir tekrarını kaldırır.
            if len(key) >= 50 and key in seen:
                continue
            if len(key) >= 50:
                seen.add(key)
            cleaned.append(sentence)
        return "\n\n".join(cleaned)

    @classmethod
    def _has_excessive_repetition(cls, text: str) -> bool:
        """Eski cache kaydının bozuk tekrar içerip içermediğini denetler."""
        return cls._remove_repeated_sentences(text) != (text or "").strip()

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
        message: Optional[str] = None,
    ) -> LegalAnswer:
        """Doğrudan hüküm yoksa yakın kanunları cevap gibi göstermeden sunar."""
        nearby: List[str] = []
        for source in sources[:3]:
            law_number = str(source.get("law_number") or "").strip()
            law = str(source.get("law_name") or law_number or "Bilinmeyen kanun").strip()
            if law_number and law_number != "unknown":
                law = re.sub(rf"^{re.escape(law_number)}[_\s-]*", "", law).strip()
                law = f"{law_number} sayılı {law}"
            article = str(source.get("article_number") or source.get("article_no") or "-")
            label = str(source.get("label") or "")
            nearby.append(f"- {law} — Madde {article} [{label}]")
        answer = message or (
            "Sağlanan kaynaklarda sorunuzun doğrudan cevabını doğrulayacak açık bir hüküm "
            "bulunamadı. Aşağıdaki kaynaklar konuya yakın görünmektedir; ancak bunlar sorunun "
            "kesin cevabı olarak değerlendirilmemelidir."
        )
        if nearby:
            answer += "\n\nYakın kaynaklar:\n" + "\n".join(nearby)
        return LegalAnswer(
            answer=answer,
            sources=sources,
            citations=render_citations(sources),
            grounded=False,
            retrieval_ms=round(retrieval_ms, 3),
            generation_ms=round(generation_ms, 3),
            total_ms=round(total_ms, 3),
            refusal_reason="direct_answer_not_found_nearby_sources_shown",
            retrieval_plan=plan_name,
            retrieval_plan_reason=plan_reason,
            draft_answer=draft_answer,
        )

    @staticmethod
    def _partial_evidence_answer(
        text: str,
        sources: List[Dict[str, object]],
        *,
        total_ms: float,
        retrieval_ms: float,
        plan_name: str,
        plan_reason: str,
    ) -> LegalAnswer:
        """Eksik cevapta doğrulanmış kısmı, ilgisiz yakın kaynak eklemeden sunar."""
        return LegalAnswer(
            answer=text,
            sources=sources,
            citations=render_citations(sources),
            grounded=False,
            retrieval_ms=round(retrieval_ms, 3),
            total_ms=round(total_ms, 3),
            refusal_reason="partial_evidence_missing_required_date",
            retrieval_plan=plan_name,
            retrieval_plan_reason=plan_reason,
        )

    @staticmethod
    def _fallback_citation_label(question: str, sources: List[Dict[str, object]]) -> str:
        """Prefer an explicitly requested law/article over raw reranker rank."""
        from RAG.retriever.query_metadata import get_query_metadata_extractor

        filters = get_query_metadata_extractor().extract(question)
        for source in sources:
            law_ok = not filters.get("law_number") or str(source.get("law_number")) == str(filters["law_number"])
            article_ok = not filters.get("article_no") or str(source.get("article_number")) == str(filters["article_no"])
            if law_ok and article_ok:
                return str(source["label"])
        return str(sources[0]["label"])

    @staticmethod
    def _best_article_lookup_source(question: str, sources: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
        """Pick a directly evidenced article for 'hangi maddede' lookups.

        This query type has a short, factual output.  Letting a generative
        model choose among five neighbouring articles produced a false answer
        for "vergi mahremiyeti" even though Madde 5 was in the evidence.  A
        lexical evidence check is safer and transparent for this narrow case.
        """
        ignored = {
            "kanun", "kanununda", "kanunu", "madde", "maddede", "hangi", "nedir",
            "duzenlenmistir", "duzenlenir", "gore", "sayili", "vergi", "usul",
        }
        terms = [term for term in tokenize(fold_turkish(question), min_len=4) if term not in ignored]
        if not terms:
            return None
        best: Optional[Dict[str, object]] = None
        best_score = 0
        for source in sources:
            # Kaynak metadatası tüm pasajı taşımaz; eşleşen metin chunk kimliğiyle alınır.
            chunk_id = str(source.get("chunk_id") or "")
            haystack = fold_turkish(str(source.get("text") or "")).casefold()
            # `text` alanı yalnız bu iç seçim adımı için aşağıda eklenir.
            score = sum(term[:6] in haystack for term in terms)
            # Devam/dipnot chunk'ı önceki metadata ile sonraki madde başlığını taşıyabilir.
            # Bu nedenle gerçek madde başlangıcı kesin öncelik alır.
            article = re.escape(str(source.get("article_number") or ""))
            if article and re.match(rf"\s*madde\s+{article}\s*[–-]", haystack):
                score += 10
            if score > best_score:
                best, best_score = source, score
        return best if best_score else None

    def nearby_sources(
        self,
        question: str,
        *,
        top_k: int = 5,
        message: Optional[str] = None,
        conversation_memory: Optional[str] = None,
    ) -> LegalAnswer:
        """Belirsiz soruda yakın kaynakları LLM ile, yeni hukukî sonuç üretmeden düzenler."""
        started = perf_counter()
        question = " ".join((question or "").split())
        if not question:
            return self._refusal("empty_question")

        from RAG.retriever.query_router import choose_query_plan

        plan = choose_query_plan(question)
        retrieval_started = perf_counter()
        results = retrieve(
            question,
            top_k=top_k,
            mode=plan.mode,
            use_prf=plan.use_prf,
            use_reranker=plan.use_reranker,
            use_graph=plan.use_graph,
        )
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        context = build_context(results)
        if not context.sources:
            return self._refusal("no_retrieved_evidence", (perf_counter() - started) * 1000)

        # İki tarihli mahkeme karşılaştırmasında karar tarihi ile yürürlük
        # tarihini birbirine karıştırmak ciddi bir hukukî hata üretir. İki
        # tarih açık kanıtla gelmediyse model hesap yapmaz; yakın maddeler
        # yalnız yön gösterici olarak sunulur.
        from RAG.retriever.query_metadata import get_query_metadata_extractor

        requested_law = str(get_query_metadata_extractor().extract(question).get("law_number") or "")
        relevant_sources = [
            source for source in context.sources
            if not requested_law or str(source.get("law_number") or "") == requested_law
        ] or context.sources
        if court_date_comparison_is_incomplete(question, relevant_sources):
            partial_answer = incomplete_court_date_comparison_answer(question, relevant_sources)
            if partial_answer:
                return self._partial_evidence_answer(
                    partial_answer, relevant_sources,
                    total_ms=(perf_counter() - started) * 1000,
                    retrieval_ms=retrieval_ms,
                    plan_name=plan.name,
                    plan_reason=plan.rationale,
                )
            return self._nearby_evidence_refusal(
                relevant_sources,
                total_ms=(perf_counter() - started) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                plan_name=plan.name,
                plan_reason=plan.rationale,
                draft_answer="",
                message=partial_answer or (
                    "Değişiklik cetvelinde yer alan tarih bulunmuştur; ancak Anayasa Mahkemesi "
                    "kararının yürürlüğe giriş tarihi kaynaklarda açıkça doğrulanamadığı için "
                    "zaman farkı hesaplanamaz."
                ),
            )
        fallback = self._nearby_evidence_refusal(
            context.sources,
            total_ms=(perf_counter() - started) * 1000,
            retrieval_ms=retrieval_ms,
            generation_ms=0.0,
            plan_name=plan.name,
            plan_reason=plan.rationale,
            draft_answer="",
            message=message,
        )
        prompt = (
            f"SORU:\n{question}\n\n"
            f"KULLANICIYA MUTLAKA BİLDİRİLECEK DURUM:\n{message or fallback.answer}\n\n"
            f"ÖNCEKİ KONUŞMA BELLEĞİ:\n{conversation_memory or '(yok)'}\n\n"
            f"YAKIN HUKUKÎ BAĞLAM:\n{context.text}\n\nDÜZENLENMİŞ CEVAP:"
        )
        generation_started = perf_counter()
        response = self.client.generate(
            InferenceRequest(
                messages=[
                    Message(role="system", content=_NEARBY_SOURCES_PROMPT + _CONVERSATION_MEMORY_RULE),
                    Message(role="user", content=prompt),
                ],
                temperature=agent_config.temperature,
                top_p=agent_config.top_p,
                max_tokens=agent_config.max_tokens,
            )
        )
        generation_ms = (perf_counter() - generation_started) * 1000
        if not response.success or not response.text.strip() or self._model_abstained(response.text):
            return fallback
        cited, invalid = validate_citations(response.text, context.sources)
        if not cited or invalid:
            return fallback
        # Etiketler geçerli olsa bile bu yol doğrudan hüküm kanıtlamaz. Bu nedenle
        # sonuçlar görünür kaynaklarla sunulur, fakat grounded işareti False kalır.
        return LegalAnswer(
            answer=response.text.strip(),
            sources=context.sources,
            citations=render_citations(context.sources),
            grounded=False,
            retrieval_ms=round(retrieval_ms, 3),
            generation_ms=round(generation_ms, 3),
            total_ms=round((perf_counter() - started) * 1000, 3),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            tokens_per_second=(
                round(response.completion_tokens / (generation_ms / 1000), 3)
                if response.completion_tokens and generation_ms > 0
                else None
            ),
            refusal_reason="direct_answer_not_found_nearby_sources_shown",
            retrieval_plan=plan.name,
            retrieval_plan_reason=plan.rationale,
        )

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        use_cache: bool = True,
        conversation_memory: Optional[str] = None,
    ) -> LegalAnswer:
        started = perf_counter()
        question = " ".join((question or "").split())
        if not question:
            return self._refusal("empty_question")

        # Aynı soru, farklı sohbet bağlamında farklı kişiye/olaya işaret edebilir.
        # Bu nedenle bellekli turda eski tekil cevap cache'i kullanılmaz.
        if use_cache and cache_config.enabled and not conversation_memory:
            hit = self.cache.get(question)
            if hit:
                cached = LegalAnswer.from_dict(
                    hit.payload,
                    cache_hit=True,
                    cache_similarity=hit.similarity,
                    total_ms=round((perf_counter() - started) * 1000, 3),
                )
                # Eski sürümden kalmış tekrar eden cevaplar kullanıcıya yeniden
                # gösterilmez; normal retrieval/generation yoluna devam edilir.
                if not self._has_excessive_repetition(cached.answer):
                    return cached

        from RAG.retriever.query_router import choose_query_plan

        plan = choose_query_plan(question)
        retrieval_started = perf_counter()
        results = retrieve(
            question,
            top_k=top_k,
            mode=plan.mode,
            use_prf=plan.use_prf,
            use_reranker=plan.use_reranker,
            use_graph=plan.use_graph,
        )
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        # Açık kanun/madde atfı yalnız ilişkili bir pasaja sessizce düşmemelidir.
        # Örneğin "CMK madde 999" sorusunda model ilgisiz maddeden ikna edici
        # ama yanlış bir cevap üretebilir.
        from RAG.retriever.query_metadata import get_query_metadata_extractor

        requested = get_query_metadata_extractor().extract(question)
        requested_law = requested.get("law_number")
        requested_article = requested.get("article_no")
        if requested_law or requested_article:
            reference_found = any(
                (not requested_law or str(item["metadata"].get("law_number")) == str(requested_law))
                and (
                    not requested_article
                    or str(item["metadata"].get("article_no") or item["metadata"].get("article_number"))
                    == str(requested_article)
                )
                for item in results
            )
            if not reference_found:
                return self._refusal(
                    "requested_legal_reference_not_found",
                    (perf_counter() - started) * 1000,
                )
        notice = service_lookup_notice(question, results)
        if notice:
            return LegalAnswer(
                answer=(
                    "Bu soru bir hizmet veya portal işlemiyle ilgilidir. Mevcut kaynaklarda "
                    "bu işlemi doğrulayacak resmî kurum rehberi bulunmadığından güvenilir "
                    "bir işlem yönlendirmesi veremem."
                ),
                grounded=False,
                retrieval_ms=round(retrieval_ms, 3),
                total_ms=round((perf_counter() - started) * 1000, 3),
                refusal_reason="official_service_source_missing",
                retrieval_plan=plan.name,
                retrieval_plan_reason=plan.rationale,
            )
        context = build_context(results)
        if not context.sources:
            return self._refusal("no_retrieved_evidence", (perf_counter() - started) * 1000)

        # ``answer`` doğrudan interaktif sohbetin ana yoludur. Bu kontrolün
        # nearby_sources ile aynı yerde bulunması, iki tarihli mahkeme
        # sorularının hangi giriş yolundan geldiğine bakılmaksızın tahminle
        # hesaplanmasını engeller.
        from RAG.retriever.query_metadata import get_query_metadata_extractor

        requested_law = str(get_query_metadata_extractor().extract(question).get("law_number") or "")
        relevant_sources = [
            source for source in context.sources
            if not requested_law or str(source.get("law_number") or "") == requested_law
        ] or context.sources
        if court_date_comparison_is_incomplete(question, relevant_sources):
            partial_answer = incomplete_court_date_comparison_answer(question, relevant_sources)
            if partial_answer:
                return self._partial_evidence_answer(
                    partial_answer, relevant_sources,
                    total_ms=(perf_counter() - started) * 1000,
                    retrieval_ms=retrieval_ms,
                    plan_name=plan.name,
                    plan_reason=plan.rationale,
                )
            return self._nearby_evidence_refusal(
                relevant_sources,
                total_ms=(perf_counter() - started) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                plan_name=plan.name,
                plan_reason=plan.rationale,
                draft_answer="",
                message=partial_answer or (
                    "Değişiklik cetvelinde yer alan tarih bulunmuştur; ancak Anayasa Mahkemesi "
                    "kararının yürürlüğe giriş tarihi kaynaklarda açıkça doğrulanamadığı için "
                    "zaman farkı hesaplanamaz."
                ),
            )

        # Deterministik madde seçiminin kanıt metnini görmesi için pasaj saklanır.
        result_text_by_id = {
            str(item["id"]): _article_scoped_text(item)
            for item in results
        }
        internal_sources = [
            {**source, "text": result_text_by_id.get(str(source.get("chunk_id")), "")}
            for source in context.sources
        ]
        direct_answer = (
            direct_cross_law_reference_answer(question, internal_sources)
            or
            direct_constitutional_annulment_answer(question, internal_sources)
            or direct_travel_duration_answer(question, internal_sources)
        )
        if direct_answer:
            return LegalAnswer(
                answer=direct_answer,
                sources=context.sources,
                citations=render_citations(context.sources),
                grounded=True,
                retrieval_ms=round(retrieval_ms, 3),
                total_ms=round((perf_counter() - started) * 1000, 3),
                retrieval_plan=plan.name,
                retrieval_plan_reason=plan.rationale,
            )
        if plan.name == "article_lookup":
            selected = self._best_article_lookup_source(question, internal_sources)
            if selected:
                public_sources = context.sources
                answer = LegalAnswer(
                    answer=(
                        f"Soruda geçen konu, {selected['law_name']} kapsamındaki "
                        f"Madde {selected['article_number']}'te düzenlenmektedir. [{selected['label']}]"
                    ),
                    sources=public_sources,
                    citations=render_citations(public_sources),
                    grounded=True,
                    retrieval_ms=round(retrieval_ms, 3),
                    total_ms=round((perf_counter() - started) * 1000, 3),
                    retrieval_plan=plan.name,
                    retrieval_plan_reason=plan.rationale,
                )
                if use_cache and cache_config.enabled:
                    self.cache.put(question, answer.to_dict())
                return answer

        prompt = (
            f"SORU:\n{question}\n\n"
            f"ÖNCEKİ KONUŞMA BELLEĞİ:\n{conversation_memory or '(yok)'}\n\n"
            f"HUKUKİ BAĞLAM:\n{context.text}\n\nCEVAP:"
        )
        generation_started = perf_counter()
        response = self.client.generate(
            InferenceRequest(
                messages=[Message(role="system", content=_SYSTEM_PROMPT + _REFERENCE_DOCUMENT_RULE + _CONVERSATION_MEMORY_RULE), Message(role="user", content=prompt)],
                temperature=agent_config.temperature,
                top_p=agent_config.top_p,
                max_tokens=agent_config.max_tokens,
            )
        )
        generation_ms = (perf_counter() - generation_started) * 1000
        if not response.success or not response.text.strip():
            return self._refusal("generation_unavailable", (perf_counter() - started) * 1000)

        response.text = self._remove_repeated_sentences(response.text)

        # Model bazen uzun bir tahmin yaptıktan sonra kanıtın yetersiz olduğunu da
        # söyler. Böyle bir metne otomatik [S1] eklemek yanlış bir grounded sonucu
        # üretir; taslak saklanır ve yakın kanunlar yalnız referans olarak gösterilir.
        if self._model_abstained(response.text):
            return self._nearby_evidence_refusal(
                context.sources,
                total_ms=(perf_counter() - started) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                plan_name=plan.name,
                plan_reason=plan.rationale,
                draft_answer=response.text.strip(),
            )

        grounded, invalid = validate_citations(response.text, context.sources)
        if invalid:
            return self._refusal("invalid_citation", (perf_counter() - started) * 1000, response.text.strip())
        if not grounded:
            # Kaynak etiketi olmayan metin kanıtlanmış cevap değildir. Modelin
            # taslağı saklanır; kullanıcıya yalnız yakın, açık kaynaklar sunulur.
            return self._nearby_evidence_refusal(
                context.sources,
                total_ms=(perf_counter() - started) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                plan_name=plan.name,
                plan_reason=plan.rationale,
                draft_answer=response.text.strip(),
            )

        tokens_per_second = None
        if response.completion_tokens and generation_ms > 0:
            tokens_per_second = round(response.completion_tokens / (generation_ms / 1000), 3)
        timings = response.server_timings or {}
        ttft_ms = timings.get("prompt_ms")
        answer = LegalAnswer(
            answer=response.text.strip(),
            sources=context.sources,
            citations=render_citations(context.sources),
            grounded=True,
            retrieval_ms=round(retrieval_ms, 3),
            generation_ms=round(generation_ms, 3),
            total_ms=round((perf_counter() - started) * 1000, 3),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            tokens_per_second=tokens_per_second,
            ttft_ms=round(float(ttft_ms), 3) if isinstance(ttft_ms, (int, float)) else None,
            retrieval_plan=plan.name,
            retrieval_plan_reason=plan.rationale,
        )
        if use_cache and cache_config.enabled and not conversation_memory:
            self.cache.put(question, answer.to_dict())
        return answer
