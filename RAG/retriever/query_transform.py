"""Local-LLM query transformation with a deterministic safe fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from RAG.configuration.rag_config_loader import query_transform_config


@dataclass(frozen=True)
class QueryTransform:
    original: str
    queries: List[str]
    intent: str = "unknown"
    used_llm: bool = False
    llm_queries: List[str] | None = None


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_COMMON_CORRECTIONS = (
    # Gündelik yazım biçimleri aramada belgelerdeki resmî karşılıkları bulamayabilir.
    (r"\bnede\b", "nerede"),
    (r"\bnerden\b", "nereden"),
    (r"\bnasil\b", "nasıl"),
    (r"\bsorgulabilir\b", "sorgulanabilir"),
    (r"\bvergi\s+borcum\b", "vergi borcumu"),
    (r"\bcalinirsa\b", "çalınırsa"),
    (r"\bhirsiz\b", "hırsız"),
)
_LEGAL_ABBREVIATIONS = (
    (r"\bcmk\b", "Ceza Muhakemesi Kanunu"),
    (r"\btck\b", "Türk Ceza Kanunu"),
    (r"\bkvkk\b", "Kişisel Verilerin Korunması Kanunu"),
    (r"\bvuk\b", "Vergi Usul Kanunu"),
)
_SYSTEM_PROMPT = """Sen Türk hukukunda arama sorgusu dönüştürücüsün.
Sadece şu geçerli JSON biçiminde cevap ver: {\"queries\":[\"sorgu 1\",\"sorgu 2\"],\"intent\":\"kısa niyet\"}.
"queries" dizisi boş OLAMAZ ve iki kısa Türkçe arama ifadesi içermelidir.
Anlamı aynen koru; günlük bir kelimenin eşdeğer resmî hukuk terimini kullan.
Özgün soruyu kısaltma ve yazım hatasını tekrar etme; her öneri tam, aramaya
uygun ve özgün sorudan en az aynı bilgi düzeyinde olmalıdır.
Soruyu cevaplama. Kanun/madde numarası, tarih, kişi, olay veya hukukî sonuç
uydurma; yalnızca soruda geçen sayıları kullan. Bir maddenin neyi düzenlediği
sorusunu değişiklik, yürürlük veya yürürlükten kalkma sorusuna çevirme."""


def _clean_queries(original: str, values: object) -> List[str]:
    """Validate untrusted model output and always retain the original query."""
    unique = [original.strip()]
    if not isinstance(values, list):
        return unique
    for value in values:
        if not isinstance(value, str):
            continue
        query = " ".join(value.split()).strip()
        if not query or len(query) > query_transform_config.max_query_chars:
            continue
        # Dönüşüm, kullanıcının belirtmediği kanun veya madde numarasını eklememelidir.
        original_numbers = set(re.findall(r"\b\d{2,8}\b", original))
        new_numbers = set(re.findall(r"\b\d{2,8}\b", query)) - original_numbers
        if new_numbers:
            continue
        if query.casefold() not in {item.casefold() for item in unique}:
            unique.append(query)
        if len(unique) >= query_transform_config.max_queries:
            break
    return unique


def _deterministic_queries(original: str) -> List[str]:
    """Yaygın Türkçe yazım hatalarını ve hukuk kısaltmalarını hızlıca normalize eder.

    Bu katman yeni kanun, madde veya olay bilgisi eklemez. Özgün soru ilk sırada
    tutulur; düzeltilmiş biçim yalnız recall artırmak için ikinci arama olur.
    """
    corrected = original
    for pattern, replacement in _COMMON_CORRECTIONS:
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    abbreviation_expanded = corrected
    for pattern, replacement in _LEGAL_ABBREVIATIONS:
        abbreviation_expanded = re.sub(pattern, replacement, abbreviation_expanded, flags=re.IGNORECASE)
    return _clean_queries(original, [corrected, abbreviation_expanded])


def _is_useful_llm_variant(original: str, candidate: str) -> bool:
    """Küçük dönüşüm modelinin kısaltılmış veya bozuk önerilerini eler."""
    normalized_candidate = candidate.casefold()
    # Model, düzeltilmesi bilinen bir yazım hatasını yeniden üretmişse kurallı
    # varyant daha güvenilirdir; bu öneri arama kalitesini düşürmemelidir.
    if any(re.search(pattern, normalized_candidate, flags=re.IGNORECASE) for pattern, _ in _COMMON_CORRECTIONS):
        return False
    original_terms = set(re.findall(r"[\wçğıöşü]+", original.casefold()))
    candidate_terms = set(re.findall(r"[\wçğıöşü]+", normalized_candidate))
    # “vergi nede” gibi özgün sorunun yalnızca kesilmiş biçimi yeni anlam veya
    # recall kazandırmaz. Tamamen yeni, anlamlı bir hukuk terimi içermelidir.
    if candidate_terms and candidate_terms.issubset(original_terms) and len(candidate) < len(original) * 0.9:
        return False
    return len(candidate_terms) >= 2


@lru_cache(maxsize=256)
def _transform_cached(original: str) -> QueryTransform:
    """Transform a normalized query once per running process.

    Interactive legal research commonly repeats or refines the same question.
    Caching means those repeats retain the accuracy benefit without another
    local-LLM request.
    """

    # Açık hukukî atıflar zaten kesindir. LLM çağrısı gecikme ekler ve
    # deterministik kanun/madde filtresini yalnız zayıflatabilir.
    from RAG.retriever.query_metadata import get_query_metadata_extractor

    if get_query_metadata_extractor().extract(original):
        return QueryTransform(original=original, queries=[original])

    deterministic = _deterministic_queries(original)
    # Ayrı dönüşüm modeli çalışmıyorsa da sorgu düzeltmesi aktif kalır. Böylece
    # RAG'ın çalışması ikinci bir LLM servisine bağımlı olmaz ve gecikme eklenmez.
    if not query_transform_config.use_llm:
        return QueryTransform(
            original=original,
            queries=deterministic,
            intent="deterministic_rewrite" if len(deterministic) > 1 else "original",
            llm_queries=[],
        )

    try:
        from Inference.client.inference_request import InferenceRequest, Message
        from Inference.client.llama_client import LlamaClient

        response = LlamaClient(
            base_url=query_transform_config.base_url,
            timeout=query_transform_config.timeout_seconds,
        ).generate(
            InferenceRequest(
                messages=[Message(role="system", content=_SYSTEM_PROMPT), Message(role="user", content=original)],
                temperature=0.0,
                top_p=1.0,
                max_tokens=query_transform_config.max_tokens,
            )
        )
        if not response.success:
            return QueryTransform(original=original, queries=deterministic, intent="deterministic_fallback")
        match = _JSON_RE.search(response.text)
        if not match:
            return QueryTransform(original=original, queries=deterministic, intent="deterministic_fallback")
        data = json.loads(match.group(0))
        raw_llm_queries = [
            item
            for item in _clean_queries(original, data.get("queries"))[1:]
            if _is_useful_llm_variant(original, item)
        ]
        queries = _clean_queries(original, [*deterministic[1:], *raw_llm_queries])
        accepted_llm_queries = [item for item in raw_llm_queries if item in queries]
        return QueryTransform(
            original=original,
            queries=queries,
            intent=str(data.get("intent", "unknown"))[:80],
            used_llm=bool(accepted_llm_queries),
            llm_queries=accepted_llm_queries,
        )
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return QueryTransform(original=original, queries=deterministic, intent="deterministic_fallback")


def transform_query(query: str, *, use_llm: bool | None = None) -> QueryTransform:
    """Return diversified searches; optionally force the non-LLM path.

    Katman sözleşmesiyle çalışan retrieval çağrılarında soru dönüştürme modeli
    de çalıştırılmaz. Kural tabanlı yazım düzeltmesi yine korunur.
    """
    original = " ".join((query or "").split()).strip()
    if not original or not query_transform_config.enabled:
        return QueryTransform(original=original, queries=[original] if original else [])
    if use_llm is False:
        deterministic = _deterministic_queries(original)
        return QueryTransform(
            original=original,
            queries=deterministic,
            intent="deterministic_rewrite" if len(deterministic) > 1 else "original",
            llm_queries=[],
        )
    return _transform_cached(original)
