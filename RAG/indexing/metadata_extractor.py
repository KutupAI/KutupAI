"""
metadata_extractor.py
------------------------
يربط كل RawChunk (الناتج من chunker.py) بمعلوماته الوصفية القانونية:
اسم القانون، رقم المادة، تاريخ النفاذ.

المصدر الأساسي لهذه المعلومات (اسم القانون، تاريخ النفاذ) هو ملف
metadata مرافق لكل مستند مصدر (مثال: kanun_5651.meta.json) بجانب
ملف النص نفسه. رقم المادة يأتي مباشرة من RawChunk (استُخرج بالفعل
أثناء التقطيع في chunker.py).

شكل ملف الـ metadata المرافق المتوقع (kanun_5651.meta.json):
{
    "law_name": "5651 Sayılı Kanun",
    "law_number": "5651",
    "effective_date": "2007-05-23"
}
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from RAG.indexing.chunker import RawChunk


@dataclass
class EnrichedChunk:
    """جزء نصي مع كامل بياناته الوصفية - جاهز لتوليد الـ embedding والحفظ."""
    text: str
    metadata: Dict[str, Any]


def _load_source_metadata(source_file: Path) -> Dict[str, Any]:
    """
    تحميل ملف الـ metadata المرافق للمستند المصدر إن وُجد.
    إذا لم يوجد، تُرجَع قيم افتراضية فارغة بدل فشل العملية بالكامل.
    """
    meta_path = source_file.with_suffix(".meta.json")
    if not meta_path.exists():
        return {
            "law_name": source_file.stem,
            "law_number": "unknown",
            "effective_date": "unknown",
        }

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def enrich_chunks(
    raw_chunks: list[RawChunk],
    source_file: Path,
) -> list[EnrichedChunk]:
    """
    نقطة الدخول الرئيسية: تحويل قائمة RawChunk إلى EnrichedChunk بإضافة
    كامل الـ metadata اللازمة للاستشهاد القانوني لاحقًا من writer_agent.

    Args:
        raw_chunks: نتيجة chunker.chunk_document().
        source_file: مسار ملف النص المصدر (لتحديد ملف الـ metadata المرافق).

    Returns:
        قائمة EnrichedChunk جاهزة لتمريرها إلى indexer.py.
    """
    source_meta = _load_source_metadata(source_file)

    enriched: list[EnrichedChunk] = []
    for chunk in raw_chunks:
        metadata = {
            "law_name": source_meta.get("law_name", "unknown"),
            "law_number": source_meta.get("law_number", "unknown"),
            "effective_date": source_meta.get("effective_date", "unknown"),
            "article_number": chunk.article_number or "unknown",
            "part_index": chunk.part_index,
            "source_file": source_file.name,
        }
        enriched.append(EnrichedChunk(text=chunk.text, metadata=metadata))

    return enriched
