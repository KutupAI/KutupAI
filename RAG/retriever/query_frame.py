"""Sorguyu retrieval için küçük ve denetlenebilir bir kanıt planına dönüştürür."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from RAG.retriever.query_metadata import (
    QueryIntent,
    _coordinated_law_numbers,
    get_query_metadata_extractor,
)
from RAG.retriever.text_utils import fold_turkish, tokenize


EvidenceKind = Literal["law", "article", "amendment", "court", "duration"]


@dataclass(frozen=True)
class EvidenceSlot:
    """Sorunun cevaplanması için gereken tek, açık kanıt parçası."""

    key: str
    kind: EvidenceKind
    law_number: str | None = None
    article_no: str | None = None
    amending_number: str | None = None


@dataclass(frozen=True)
class QueryFrame:
    """Router, SQL indeks ve reranker'ın ortak soru temsili."""

    intent: QueryIntent
    kind: str
    law_numbers: tuple[str, ...]
    article_numbers: tuple[str, ...]
    target_law_numbers: tuple[str, ...]
    amending_numbers: tuple[str, ...]
    strict_law_numbers: tuple[str, ...]
    needs_amendment_evidence: bool
    needs_multiple_evidence: bool
    slots: tuple[EvidenceSlot, ...]


def build_query_frame(question: str, *, extractor: object | None = None) -> QueryFrame:
    """Yeni kelimeler için sadece sözlüğe değil, atıf ve kanıt yapısına bakar."""
    extractor = extractor or get_query_metadata_extractor()
    intent = extractor.extract_intent(question)
    normalized = fold_turkish(question).casefold()
    amendment_terms = (
        "degisiklik", "degistir", "etkile", "iptal", "mulga", "mevzuat tablosu", "mevzuat listesi",
        "degisiklik tablosu", "degisiklik cetveli", "khk", "anayasa mahkemesi", "yuksek mahkeme",
        "gecersiz kil", "iptal karari", "kanun yoluyla", "sayili kanun ile",
        "ile yapilan duzenleme", "duzenlem", "yururluge giris",
    )
    comparison_terms = ("karsilastir", "karsilastirma", "arasindaki fark", "fark nedir", "ayrimi", "kiyasla")
    duration_terms = ("kac saat", "kac gun", "kac ay", "ne kadar sure", "en gec")
    court_terms = ("anayasa mahkem", "yuksek mahk", "esas say", "karar say", "iptal karari")

    needs_amendment = any(term in normalized for term in amendment_terms)
    is_comparison = any(term in normalized for term in comparison_terms)
    needs_duration = any(term in normalized for term in duration_terms)
    needs_court = any(term in normalized for term in court_terms)
    laws = tuple(dict.fromkeys(intent.law_numbers))
    strict_extractor = getattr(extractor, "extract_strict_filters", None)
    strict_filters = strict_extractor(question) if callable(strict_extractor) else {}
    strict_laws = tuple(
        value for value in (str(strict_filters.get("law_number") or ""),) if value
    )
    khk_numbers = tuple(re.findall(r"\bkhk\s*[-/]?\s*(\d{2,5})\b", normalized, flags=re.IGNORECASE))
    numbered_laws = tuple(dict.fromkeys((
        *_coordinated_law_numbers(question),
        *re.findall(r"\b(\d{2,5})\s*(?:sayili|numarali)\b", normalized),
    )))
    # "2911 sayılı ... Kanunu ile 5187 sayılı ... Kanunu" gibi ifadelerde iki
    # numara da hedef kanundur; bunları değiştirici düzenleme diye yorumlamak
    # ikinci kanunun cetvel kanıtını kaybettiriyordu. Başlığı açıkça yazılmış
    # kanunlar target slotu olur, yalnız "7418 sayılı Kanun ile" gibi yalın
    # düzenleme numaraları amendment slotunda kalır.
    titled_law_matches = re.findall(
        r"\b(\d{2,5})\s+say[ıi]l[ıi]\s+([^.;,]{0,100}?)\bkanun(?:u|un|una|unun|da|dan)?\b",
        normalized,
        flags=re.IGNORECASE,
    )
    generic_title_terms = {"ile", "yapilan", "degisiklik", "degisiklikler", "kapsaminda", "uyarinca", "ek"}
    titled_laws = tuple(dict.fromkeys(
        number for number, title in titled_law_matches
        if any(token not in generic_title_terms for token in tokenize(title, min_len=3))
    ))
    # "7547 ... 4483 sayılı Kanunda" ifadesinde hedef 4483'tür.
    explicit_targets = tuple(dict.fromkeys(re.findall(
        r"\b(\d{2,5})\s+say[ıi]l[ıi]\s+kanun(?:da|nda|daki|undaki)\b",
        normalized,
        flags=re.IGNORECASE,
    )))
    # Adapter'ın eklediği hedef bilgisi ilk düzenleme numarasını geçer.
    contextual_targets = tuple(dict.fromkeys(re.findall(
        r"\bhedef\s+kanun\s*:\s*(\d{2,5})\s+say[ıi]l[ıi]\s+kanun\b",
        normalized,
        flags=re.IGNORECASE,
    )))
    inferred_targets = tuple(dict.fromkeys((*strict_laws, *titled_laws)))
    # Birden çok düzenleme numarası, açık hedef yoksa hedef sayılmaz.
    if needs_amendment and len(numbered_laws) > 1 and not (explicit_targets or contextual_targets or titled_laws):
        target_laws = ()
    else:
        target_laws = explicit_targets or contextual_targets or inferred_targets
    # Düzenlemeleri kullanıcının yazdığı sırada tut.
    candidate_amendments = (*numbered_laws, *intent.amending_law_numbers, *khk_numbers)
    # Açık hedef kanun, düzenleme numarası değildir. Hedef belirsizse bütün
    # numaralar structured tabloda aranır; SQL sonucu doğru hedefi belirler.
    amending = tuple(dict.fromkeys(
        value for value in candidate_amendments
        if not target_laws or value not in target_laws
    ))
    slots: list[EvidenceSlot] = []

    # Açık madde, genel law slotundan daha güçlü bir kanıttır.
    for article_no in intent.article_numbers:
        slots.append(EvidenceSlot(
            key=f"article:{intent.primary_law_number or 'unknown'}:{article_no}",
            kind="article", law_number=intent.primary_law_number, article_no=article_no,
        ))
    if needs_amendment:
        # Karşılaştırma sorusunda aynı değiştiren kanun birden fazla hedef
        # kanunun cetvelinde aranabilir. Her hedef için ayrı slot üretmek,
        # örneğin 2911 ve 5187'nin 7418 kayıtlarından yalnız ilkini finalde
        # tutma hatasını önler.
        amendment_targets = target_laws or tuple(
            value for value in (intent.primary_law_number,) if value
        )
        if amending:
            for target_law in amendment_targets or (None,):
                for amending_number in amending:
                    slots.append(EvidenceSlot(
                        key=f"amendment:{target_law or 'unknown'}:{amending_number}",
                        kind="amendment", law_number=target_law, amending_number=amending_number,
                    ))
        else:
            for target_law in amendment_targets or (None,):
                slots.append(EvidenceSlot("amendment", "amendment", law_number=target_law))
    if needs_court:
        slots.append(EvidenceSlot("court", "court", law_number=intent.primary_law_number))
    if needs_duration:
        slots.append(EvidenceSlot("duration", "duration", law_number=intent.primary_law_number))
    if len(laws) > 1 or is_comparison:
        for law_number in laws:
            slots.append(EvidenceSlot(f"law:{law_number}", "law", law_number=law_number))
    elif not slots and intent.primary_law_number:
        slots.append(EvidenceSlot(f"law:{intent.primary_law_number}", "law", law_number=intent.primary_law_number))

    # Aynı slotun birden çok kez eklenmesi final çeşitlendirmesini bozmaz.
    unique_slots = tuple({slot.key: slot for slot in slots}.values())
    if is_comparison:
        kind = "comparison"
    elif needs_amendment:
        kind = "amendment"
    elif len(laws) > 1 or intent.kind == "multi_law_relation":
        kind = "multi_law_relation"
    else:
        kind = intent.kind
    return QueryFrame(
        intent=intent,
        kind=kind,
        law_numbers=laws,
        article_numbers=intent.article_numbers,
        target_law_numbers=target_laws,
        amending_numbers=amending,
        strict_law_numbers=strict_laws,
        needs_amendment_evidence=needs_amendment,
        needs_multiple_evidence=len(unique_slots) > 1 or intent.needs_multiple_evidence,
        slots=unique_slots,
    )
