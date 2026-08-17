"""Choose the authoritative corpus scope before retrieval."""

from __future__ import annotations

from RAG.retriever.text_utils import fold_turkish


_AMENDMENT_TERMS = (
    "degisiklik", "degistir", "resmi gazete", "resmi gazete",
    "yururluk", "yururluge", "ek madde", "torba kanun",
)
_REGULATION_TERMS = ("yonetmelik", "teblig", "tuzuk", "genelge")


def default_source_where(query: str, explicit_where: dict | None) -> dict | None:
    """Prefer consolidated primary texts unless the user asks about changes.

    Amendment notices are valuable evidence for change-history questions but
    commonly outrank the consolidated statute for ordinary legal questions.
    Explicit caller filters always take priority.
    """
    if explicit_where:
        return explicit_where
    normalized = fold_turkish(query or "").casefold()
    if any(term in normalized for term in _AMENDMENT_TERMS):
        return None
    if any(term in normalized for term in _REGULATION_TERMS):
        return {"source_type": "regulations"}
    # Birincil corpus yürürlükteki bütünleşik kanun metinlerini tutar. Yönetmelik
    # ancak adı verildiğinde aranır; aksi halde tarihî alıntılar ana maddeyi ilk
    # sıradan düşürebilir.
    return {"source_type": "laws"}
