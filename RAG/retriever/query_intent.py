"""Small, deterministic safeguards for questions outside a legal-text corpus."""

from __future__ import annotations

import re
from typing import Iterable

from RAG.retriever.text_utils import fold_turkish
from RAG.vector_store.vector_store_interface import SearchResult


_SERVICE_LOOKUP = re.compile(
    r"\b(nerden|nerede|hangi\s+(site|uygulama|portaldan)|e\s*-?\s*devlet|"
    r"nasil\s+sorgula|sorgulayabil|"
    r"online\s+sorgu|internet(?:ten)?\s+sorgu|"
    r"resmi\s+(internet|web)|internet\s+sayfas)\b",
    re.IGNORECASE,
)
_OFFICIAL_SERVICE_MARKERS = ("e-devlet", "turkiye.gov.tr", "gib.gov.tr", "gelir idaresi", "dijital vergi")


def asks_for_service_lookup(query: str) -> bool:
    """Açıkça portal veya uygulama adımı isteyen soruları ayırt eder."""
    normalized = fold_turkish(query or "").casefold()
    return bool(_SERVICE_LOOKUP.search(normalized))


def has_official_service_evidence(results: Iterable[SearchResult]) -> bool:
    """Only allow an operational answer when its official guide is indexed."""
    for result in results:
        meta = result["metadata"]
        searchable = " ".join((
            str(result.get("text") or "")[:1000],
            str(meta.get("source_file") or ""),
            str(meta.get("source_type") or ""),
        )).casefold()
        if any(marker in searchable for marker in _OFFICIAL_SERVICE_MARKERS):
            return True
    return False


def service_lookup_notice(query: str, results: Iterable[SearchResult]) -> str | None:
    if asks_for_service_lookup(query) and not has_official_service_evidence(results):
        return (
            "Bu soru bir hizmet/portal işlemi istiyor. Mevcut corpus'ta bu işlemi "
            "doğrulayan resmî e-Devlet veya kurum rehberi bulunmadığı için aşağıdaki "
            "sonuçlar yalnızca hukukî arka plan olarak değerlendirilmelidir."
        )
    return None
