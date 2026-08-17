"""Build the OCR Agent's stable output contract (see README section
"OCR output contract"). Downstream Agents (Classification, Extraction,
RAG, Writer...) depend only on this shape — never on PaddleOCR internals.

Contract fields deliberately EXCLUDE anything semantic (question, answer,
summary, classification, extracted_data, validation, RAG, law/article
numbers). Those belong to later Agents.
"""

from __future__ import annotations

import re
from typing import Any

from Agents.ocr_agent.interfaces.signature_detector import SealDetection, SignatureDetection
from Agents.ocr_agent.models import LayoutElement, OCRTextItem, TableResult

_BLOCK_TYPES = {
    "title", "paragraph", "text", "table", "image", "header",
    "footer", "seal", "signature", "figure", "chart",
}


def _normalize_block_type(raw_type: str) -> str:
    t = (raw_type or "").strip().lower()
    if t in _BLOCK_TYPES:
        return t
    if "title" in t or "heading" in t:
        return "title"
    if "head" in t:
        return "header"
    if "foot" in t:
        return "footer"
    if "table" in t:
        return "table"
    if "seal" in t or "stamp" in t:
        return "seal"
    if "sign" in t:
        return "signature"
    if "image" in t or "figure" in t or "picture" in t:
        return "image"
    return "unknown"


def _overlaps(a: list[float], b: list[float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def build_blocks(
    text_items: list[OCRTextItem],
    layout: list[LayoutElement],
    low_confidence_threshold: float,
) -> list[dict[str, Any]]:
    from Agents.ocr_agent.core.ocr_parser import group_paragraphs, join_page_text

    blocks: list[dict[str, Any]] = []
    ordered_layout = sorted(
        layout,
        key=lambda e: (e.bounding_box.as_xyxy()[1], e.bounding_box.as_xyxy()[0]),
    )

    if ordered_layout:
        used_items: set[int] = set()
        for element in ordered_layout:
            e_box = element.bounding_box.as_xyxy()
            matched: list[OCRTextItem] = []
            matched_confidences: list[float] = []
            for idx, item in enumerate(text_items):
                if idx in used_items:
                    continue
                if _overlaps(e_box, item.bounding_box.as_xyxy()):
                    matched.append(item)
                    matched_confidences.append(item.confidence)
                    used_items.add(idx)
            text = join_page_text(matched)
            confidence = (
                sum(matched_confidences) / len(matched_confidences)
                if matched_confidences else element.confidence
            )
            block_type = _normalize_block_type(element.element_type)
            if block_type in {"table", "image"} and not text:
                blocks.append(
                    _block(block_type, "", element.bounding_box.as_xyxy(), confidence,
                           low_confidence_threshold)
                )
                continue
            if text:
                blocks.append(
                    _block(block_type, text, element.bounding_box.as_xyxy(), confidence,
                           low_confidence_threshold)
                )

        leftover = [item for idx, item in enumerate(text_items) if idx not in used_items]
        if leftover:
            for text, group in group_paragraphs(leftover):
                if not text:
                    continue
                conf = sum(i.confidence for i in group) / len(group)
                bbox = _union_bbox(group)
                blocks.append(_block("paragraph", text, bbox, conf, low_confidence_threshold))
        return blocks

    for text, group in group_paragraphs(text_items):
        if not text:
            continue
        conf = sum(i.confidence for i in group) / len(group)
        bbox = _union_bbox(group)
        blocks.append(_block("paragraph", text, bbox, conf, low_confidence_threshold))
    return blocks


def _union_bbox(items: list[OCRTextItem]) -> list[float] | None:
    xyxy = [item.bounding_box.as_xyxy() for item in items]
    if not xyxy:
        return None
    xs0, ys0, xs1, ys1 = zip(*xyxy)
    box = [min(xs0), min(ys0), max(xs1), max(ys1)]
    if max(abs(v) for v in box) <= 1.0:
        return None
    return box


def _block(
    block_type: str,
    text: str,
    bbox: list[float] | None,
    confidence: float | None,
    low_confidence_threshold: float,
) -> dict[str, Any]:
    block: dict[str, Any] = {"type": block_type, "text": text}
    if bbox is not None:
        block["bbox"] = [round(v, 2) for v in bbox]
    if confidence is not None:
        block["confidence"] = round(float(confidence), 4)
        block["uncertain"] = confidence < low_confidence_threshold
    else:
        block["uncertain"] = False
    return block


def build_table_contract(tables: list[TableResult]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for table in tables:
        rows: list[list[str]] = []
        if table.cells:
            max_row = max(c.row for c in table.cells)
            max_col = max(c.column for c in table.cells)
            grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
            for cell in table.cells:
                grid[cell.row][cell.column] = cell.text
            rows = grid
        entry: dict[str, Any] = {"rows": rows}
        if table.bounding_box is not None:
            entry["bbox"] = [round(v, 2) for v in table.bounding_box.as_xyxy()]
        if table.confidence is not None:
            entry["confidence"] = round(float(table.confidence), 4)
        out.append(entry)
    return out


def build_vision_contract(
    signature: SignatureDetection, seal: SealDetection
) -> dict[str, Any]:
    """Return only the public visual findings.

    Detection confidence and coordinates remain available on the internal
    detection objects, but are intentionally not part of the OCR result.
    """
    return {
        "signature": {
            "detected": bool(signature.detected),
            "handwritten": bool(signature.handwritten) if signature.detected else False,
        },
        "stamp": {"detected": bool(seal.detected)},
    }


def detect_language(text: str, configured_language: str) -> dict[str, Any]:
    """Conservatively identify Turkish, including OCR missing diacritics."""
    turkish_chars = set("çğıöşüÇĞİÖŞÜ")
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return {"detected": configured_language, "confidence": 0.0}
    tr_hits = sum(1 for c in letters if c in turkish_chars)
    ratio = tr_hits / len(letters)
    if ratio > 0.0:
        confidence = min(0.99, 0.5 + ratio * 10)
        return {"detected": "tr", "confidence": round(confidence, 2)}

    # RapidOCR's generic fallback can lose Turkish diacritics. This does not
    # alter OCR text; it only recognizes a document language from several
    # independent, common Turkish words in the already-extracted content.
    words = set(re.findall(r"[a-zA-Z]+", text.casefold()))
    turkish_words = {
        "adres", "alinmistir", "ancak", "basvuru", "bildirim", "bir",
        "bulunmasi", "diger", "durumunda", "edinir", "edilir", "halinde",
        "icin", "ilgili", "islem", "kadar", "kapsaminda", "menusu",
        "odeme", "olarak", "poliçe", "police", "prim", "sigorta",
        "sozlesme", "taraf", "teminat", "tuketici", "uygulanir", "ve",
        "yapilan", "yazili", "yonetmelik",
    }
    evidence = words & turkish_words
    if len(evidence) >= 3:
        return {"detected": "tr", "confidence": min(0.95, 0.55 + len(evidence) * 0.06)}
    return {"detected": "en", "confidence": 0.6}
