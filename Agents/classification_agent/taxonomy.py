"""
taxonomy.py
-------------
Canonical 18-class document taxonomy for classification_agent.


"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentClass:
    code: str          # machine-readable id, used in JSON output (document_type)
    tr_name: str        # original Turkish name from the task document
    order: int           # 1..18, matches §5 list order


DOCUMENT_CLASSES: tuple[DocumentClass, ...] = (
    DocumentClass("dilekce", "Dilekçe", 1),
    DocumentClass("basvuru_belgesi", "Başvuru Belgesi", 2),
    DocumentClass("talep_yazisi", "Talep Yazısı", 3),
    DocumentClass("sikayet_basvurusu", "Şikâyet Başvurusu", 4),
    DocumentClass("itiraz_basvurusu", "İtiraz Başvurusu", 5),
    DocumentClass("bilgi_edinme_basvurusu", "Bilgi Edinme Başvurusu", 6),
    DocumentClass("resmi_yazi", "Resmî Yazı", 7),
    DocumentClass("ust_yazi", "Üst Yazı", 8),
    DocumentClass("izin_belgesi", "İzin Belgesi", 9),
    DocumentClass("onay_belgesi", "Onay Belgesi", 10),
    DocumentClass("tutanak", "Tutanak", 11),
    DocumentClass("form", "Form", 12),
    DocumentClass("beyan_beyanname", "Beyan / Beyanname", 13),
    DocumentClass("bildirim_tebligat", "Bildirim / Tebligat", 14),
    DocumentClass("rapor", "Rapor", 15),
    DocumentClass("karar_karar_yazisi", "Karar / Karar Yazısı", 16),
    DocumentClass("sozlesme_protokol", "Sözleşme / Protokol", 17),
    DocumentClass("diger_belirsiz", "Diğer / Belirsiz", 18),
)

VALID_CODES: frozenset[str] = frozenset(c.code for c in DOCUMENT_CLASSES)

CODE_TO_TR_NAME: dict[str, str] = {c.code: c.tr_name for c in DOCUMENT_CLASSES}

UNCERTAIN_CODE = "diger_belirsiz"


def is_valid_code(code: str) -> bool:
    return code in VALID_CODES
