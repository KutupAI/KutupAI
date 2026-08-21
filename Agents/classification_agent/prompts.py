"""
prompts.py
------------
Prompt templates for classification_agent's Qwen VLM call.

Design constraints from the task document:
- section 4: Qwen must reason over image + OCR text + layout together, not
  "what is this document?" on text alone.
- section 7: output must be STRICT parse-able JSON, no free-text explanation.
- section 5: the 18-class taxonomy is a starting point, not fixed law --
  the prompt exposes it as data (from taxonomy.py) so updating the taxonomy
  file is enough; nothing here needs to change in lockstep.
"""

from __future__ import annotations

from typing import Any

from Agents.classification_agent.taxonomy import DOCUMENT_CLASSES

SYSTEM_PROMPT = """Sen bir kamu evrakı sınıflandırma uzmanısın. Görevin, sana verilen
belge görüntüsünü, OCR metnini ve (varsa) layout bilgisini birlikte
değerlendirerek belgenin hangi sınıfa ait olduğuna karar vermektir.

KURALLAR:
1. Yalnızca aşağıda listelenen sınıflardan birini seç. Listede olmayan bir
   sınıf UYDURMA.
2. Kararını sadece OCR metnine değil; başlık yerleşimi, form/tablo yapısı,
   imza alanı, kurum şablonu gibi görsel ipuçlarına da dayandır.
3. Emin değilsen veya belge birden fazla sınıfa benziyorsa, en olası sınıfı
   document_type olarak yine de bildir; ama confidence değerini düşük tut ve
   alternatives listesine diğer olası sınıfları ekle.
4. SADECE geçerli JSON döndür. Açıklama, yorum, markdown kod bloğu YOK.
5. confidence değerleri 0 ile 1 arasında olmalı ve document_type +
   alternatives içindeki tüm confidence'lar mantıklı bir dağılım oluşturmalı
   (birbirine yakın belirsiz adaylar varsa bunu confidence'a yansıt)."""


def build_taxonomy_block() -> str:
    lines = [f"{c.order}. {c.code} -- {c.tr_name}" for c in DOCUMENT_CLASSES]
    return "\n".join(lines)


def build_output_schema_block() -> str:
    return (
        "{\n"
        '  "document_type": "<yukaridaki_code_degerlerinden_biri>",\n'
        '  "confidence": <0-1 arasi float>,\n'
        '  "alternatives": [\n'
        '    {"type": "<code>", "confidence": <0-1 arasi float>},\n'
        "    ...\n"
        "  ]\n"
        "}"
    )


def build_user_prompt(
    *,
    normalized_text: str,
    ocr_confidence: float | None,
    layout_summary: str | None,
    vision_summary: str | None = None,
    top_k_alternatives: int,
) -> str:
    """Assemble the text portion of the Qwen VLM user message.

    The rendered page image (when config.send_image is True) is attached as
    a separate multimodal content part by tools.py -- this function only
    builds the text instructions + evidence block.
    """
    parts: list[str] = []
    parts.append("BELGE SINIFLARI (yalnizca bunlardan biri secilebilir):")
    parts.append(build_taxonomy_block())
    parts.append("")

    if ocr_confidence is not None:
        parts.append(f"OCR guven skoru: {ocr_confidence:.2f}")
        if ocr_confidence < 0.55:
            parts.append(
                "(OCR guveni dusuk -- metne asiri guvenme, gorsel ipuclarina "
                "daha fazla agirlik ver.)"
            )
        parts.append("")

    parts.append("OCR METNI (normalize edilmis):")
    parts.append(normalized_text.strip() or "(bos -- yalnizca gorsele dayan)")
    parts.append("")

    if layout_summary:
        parts.append("LAYOUT BILGISI:")
        parts.append(layout_summary.strip())
        parts.append("")

    if vision_summary:
        parts.append("GORSEL TESPIT SONUCLARI (OCR katmanindan):")
        parts.append(vision_summary.strip())
        parts.append(
            "(Bu bilgiyi karar verirken dikkate al -- ornegin el yazisi imza "
            "'dilekce' gibi elle yazilan turlerle, muhur/kase ise resmi "
            "kurum belgeleriyle daha cok orusur.)"
        )
        parts.append("")

    parts.append(f"En fazla {top_k_alternatives} alternatif sinif bildir.")
    parts.append("")
    parts.append("YANIT FORMATI (sadece bu JSON, baska hicbir sey yazma):")
    parts.append(build_output_schema_block())

    return "\n".join(parts)


def build_layout_summary(layout: Any) -> str | None:
    """Turn raw layout info (list of layout elements from ocr_agent, or a
    dict) into a short human-readable block for the prompt. Returns None if
    there is nothing usable -- callers should omit the layout section rather
    than send an empty block.
    """
    if not layout:
        return None

    if isinstance(layout, str):
        return layout

    if isinstance(layout, list):
        lines = []
        for item in layout:
            if isinstance(item, dict):
                el_type = item.get("element_type") or item.get("type")
                page = item.get("page_index")
                if el_type is not None:
                    lines.append(f"- {el_type}" + (f" (sayfa {page})" if page is not None else ""))
        return "\n".join(lines) if lines else None

    if isinstance(layout, dict):
        lines = [f"- {key}: {value}" for key, value in layout.items()]
        return "\n".join(lines) if lines else None

    return None


def build_vision_signal_summary(pages: Any) -> str | None:
    """Turn ocr_agent's per-page `vision` block (signature/stamp detection)
    into a short prompt-ready line. This is real signal ocr_agent already
    produces (see Documentation / ocr_agent output samples) -- distinct
    from `layout`, since ocr_agent's actual output has no `layout` key per
    page, only `vision`. Feeding this in matters most for the §9 hard case
    "el yazısı/ıslak imza bulunan belge" and for telling apart classes that
    hinge on whether a document is hand-signed vs. stamped/official.
    """
    if not isinstance(pages, list):
        return None

    lines: list[str] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        vision = page.get("vision")
        if not isinstance(vision, dict):
            continue

        signature = vision.get("signature") or {}
        stamp = vision.get("stamp") or {}
        page_no = page.get("page_number")
        prefix = f"Sayfa {page_no}: " if page_no is not None else ""

        if signature.get("detected"):
            kind = "el yazısı imza" if signature.get("handwritten") else "imza"
            lines.append(f"{prefix}{kind} tespit edildi.")
        if stamp.get("detected"):
            lines.append(f"{prefix}mühür/kaşe tespit edildi.")

    return "\n".join(lines) if lines else None