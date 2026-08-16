"""
hard_cases.py
---------------
The 10 hard test scenarios required by the task document, section 9
(exact list, translated tag codes only -- Turkish wording kept verbatim
from the source document so nothing gets lost in translation):

  1. Yamuk/deskew gerektiren PDF
  2. Dusuk cozunurluklu veya bulanik tarama
  3. OCR karakter hatalari bulunan belge
  4. El yazisi/islak imza bulunan belge
  5. Cok kisa belge
  6. Cok uzun belge
  7. Benzer iki belge turu (or. dilekce-talep yazisi)
  8. Farkli kurumlarin farkli sablonlari
  9. Eksik sayfali veya eksik alanli belge
  10. Birden fazla belge turune benzeyen belirsiz ornek

Manifest rows tag which scenario(s) a document exercises via
LabeledDocument.hard_case_tags (codes below), so evaluation can report
accuracy/F1 broken down per scenario -- required for the "Zor test
senaryolarinin sonuclari" deliverable in section 11.
"""

from __future__ import annotations

from dataclasses import dataclass

from Agents.classification_agent.dataset.schema import LabeledDocument


@dataclass(frozen=True)
class HardCase:
    code: str
    tr_description: str


HARD_CASES: tuple[HardCase, ...] = (
    HardCase("skewed_pdf", "Yamuk/deskew gerektiren PDF"),
    HardCase("low_res_blurry", "Düşük çözünürlüklü veya bulanık tarama"),
    HardCase("ocr_char_errors", "OCR karakter hataları bulunan belge"),
    HardCase("handwriting_signature", "El yazısı/ıslak imza bulunan belge"),
    HardCase("very_short_doc", "Çok kısa belge"),
    HardCase("very_long_doc", "Çok uzun belge"),
    HardCase("similar_pair", "Benzer iki belge türü (ör. dilekçe–talep yazısı)"),
    HardCase("different_templates", "Farklı kurumların farklı şablonları"),
    HardCase("missing_pages_fields", "Eksik sayfalı veya eksik alanlı belge"),
    HardCase("ambiguous_multi_type", "Birden fazla belge türüne benzeyen belirsiz örnek"),
)

VALID_HARD_CASE_CODES = frozenset(h.code for h in HARD_CASES)


def breakdown_by_hard_case(
    records: list[LabeledDocument],
    predictions: dict[str, str],
) -> dict[str, dict]:
    """predictions: document_id -> predicted document_type.

    Returns per-scenario accuracy + n, only over documents tagged with that
    scenario. A scenario with 0 tagged documents is reported with n=0
    rather than omitted, so an untested scenario stays visible in the
    §11 deliverable rather than silently missing.
    """
    from Agents.classification_agent.evaluation.metrics import accuracy

    result: dict[str, dict] = {}
    for case in HARD_CASES:
        tagged = [r for r in records if case.code in r.hard_case_tags and r.label]
        y_true = [r.label for r in tagged]
        y_pred = [predictions.get(r.document_id, "") for r in tagged]
        result[case.code] = {
            "description": case.tr_description,
            "n": len(tagged),
            "accuracy": accuracy(y_true, y_pred) if tagged else None,
        }
    return result
