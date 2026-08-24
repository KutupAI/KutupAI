"""Choose the authoritative corpus scope before retrieval."""

from __future__ import annotations

from RAG.retriever.text_utils import fold_turkish


_AMENDMENT_TERMS = (
    "degisiklik", "degistir", "resmi gazete", "resmi gazete",
    "yururluk", "yururluge", "ek madde", "torba kanun",
)
# ``tebliğ`` Türkçede hem bir düzenleme türü hem de "tebliğ edilmek" fiilidir.
# İkinci kullanım çok yaygındır; tek başına corpus'u yönetmeliklere kapatmak
# Kabahatler Kanunu gibi doğrudan kanun sorularını yanlış havuza gönderiyordu.
# Bu yüzden yalnız belge türünü açıkça anlatan biçimler kaynak filtresi uygular.
_REGULATION_TERMS = (
    "yonetmelik", "yonetmeligi", "tebligi", "teblig numarasi", "tuzuk", "genelge",
)
_REFERENCE_DOCUMENT_TERMS = (
    "dilekce ornegi", "basvuru formu", "sikayet dilekcesi", "belge turu",
    "bu belge", "bu dokuman", "bu dosya", "tutanak ornegi", "ust yazi",
    "resmi yazi ornegi", "protokol ornegi", "sozlesme ornegi", "form nasil",
    "dilekce nasil", "dilekce hazirla", "sikayet nasil", "basvuru nasil",
    "form doldur", "tutanak nasil", "ust yazi nasil", "resmi yazi nasil",
)
_LEGAL_QUESTION_TERMS = (
    "kanun", "madde", "hukuk", "hak", "ceza", "yukumluluk", "zorunlu",
    "sure", "usulsuzluk", "mevzuat", "duzenlen", "sayili", "kvkk", "cmk", "tck",
)


def default_source_where(query: str, explicit_where: dict | None) -> dict | None:
    """Prefer consolidated primary texts unless the user asks about changes.

    Amendment notices are valuable evidence for change-history questions but
    commonly outrank the consolidated statute for ordinary legal questions.
    Explicit caller filters always take priority.
    """
    if explicit_where:
        return explicit_where
    normalized = fold_turkish(query or "").casefold()
    asks_for_reference = any(term in normalized for term in _REFERENCE_DOCUMENT_TERMS)
    asks_for_legal_basis = any(term in normalized for term in _LEGAL_QUESTION_TERMS)
    if asks_for_reference and asks_for_legal_basis:
        # Kullanıcı hem belge örneği/pratiğini hem de dayanak hükmü ister.
        # Tek kaynağa düşürmek yerine iki corpus aynı aday havuzunda aranır.
        return {"source_type": {"$in": ["laws", "reference_docs"]}}
    if asks_for_reference:
        return {"source_type": "reference_docs"}
    if any(term in normalized for term in _AMENDMENT_TERMS):
        return None
    # "Kararın tebliği tarihinden itibaren" de, başlığı "... Tebliği" olan
    # bir düzenleme de aynı yüzey biçimini taşır. Belirsiz durumda yalnız
    # regulations'a kapanmak yerine iki corpus'u serbest bırakmak güvenlidir.
    if "teblig" in normalized:
        return None
    if any(term in normalized for term in _REGULATION_TERMS):
        return {"source_type": "regulations"}
    # Birincil corpus yürürlükteki bütünleşik kanun metinlerini tutar. Yönetmelik
    # ancak adı verildiğinde aranır; aksi halde tarihî alıntılar ana maddeyi ilk
    # sıradan düşürebilir.
    return {"source_type": "laws"}
