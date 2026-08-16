"""
chunker.py
------------
تقطيع النصوص القانونية باحترام بنية المادة/الفقرة (madde/fıkra) كوحدة
أساسية، بدلاً من التقطيع العشوائي بحجم ثابت.

المنطق:
1. تُقسَّم الوثيقة أولًا حسب "المادة" (Madde) باستخدام تعبير نمطي يتعرف
   على الصيغة الشائعة في النصوص القانونية التركية (مثال: "MADDE 5-").
2. إذا كانت المادة نفسها طويلة جدًا (أطول من max_chunk_size_chars)،
   تُقسَّم داخليًا مع الحفاظ على تداخل (overlap) بسيط بين الأجزاء.
"""

import re
from dataclasses import dataclass
from typing import List

from RAG.configuration.rag_config_loader import chunking_config


@dataclass
class RawChunk:
    """جزء نصي خام قبل ربطه بالـ metadata."""
    text: str
    article_number: str  # رقم المادة كما استُخرج من النص، أو "" إذا غير معروف
    part_index: int      # ترتيب هذا الجزء داخل نفس المادة (0 إذا لم تُقسَّم)


# نمط للتعرف على بداية مادة جديدة، مثال: "MADDE 5-" أو "Madde 12 –"
_ARTICLE_PATTERN = re.compile(r"(MADDE|Madde)\s+(\d+)\s*[-–—]", re.UNICODE)


def _split_by_article(full_text: str) -> List[RawChunk]:
    """تقسيم النص الكامل إلى مواد (madde) بالاعتماد على _ARTICLE_PATTERN."""
    matches = list(_ARTICLE_PATTERN.finditer(full_text))

    if not matches:
        # لا يوجد تقسيم مواد واضح بالنص - يُعامل كوحدة واحدة
        return [RawChunk(text=full_text.strip(), article_number="", part_index=0)]

    chunks: List[RawChunk] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        article_text = full_text[start:end].strip()
        article_number = match.group(2)
        chunks.append(RawChunk(text=article_text, article_number=article_number, part_index=0))

    return chunks


def _split_long_chunk(chunk: RawChunk, max_size: int, overlap: int) -> List[RawChunk]:
    """تقسيم مادة طويلة إلى أجزاء أصغر مع الحفاظ على تداخل بينها."""
    text = chunk.text
    if len(text) <= max_size:
        return [chunk]

    parts: List[RawChunk] = []
    start = 0
    part_index = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        part_text = text[start:end].strip()
        if part_text:
            parts.append(
                RawChunk(
                    text=part_text,
                    article_number=chunk.article_number,
                    part_index=part_index,
                )
            )
            part_index += 1
        if end == len(text):
            break
        start = end - overlap  # تراجع بمقدار الـ overlap لبداية الجزء التالي

    return parts


def chunk_document(full_text: str) -> List[RawChunk]:
    """
    نقطة الدخول الرئيسية: تقطيع نص وثيقة قانونية كاملة إلى قائمة RawChunk.

    Args:
        full_text: النص الكامل للوثيقة (قانون/لائحة) بعد قراءته من الملف.

    Returns:
        قائمة RawChunk جاهزة لتمريرها إلى metadata_extractor.py.
    """
    article_chunks = _split_by_article(full_text)

    final_chunks: List[RawChunk] = []
    for chunk in article_chunks:
        final_chunks.extend(
            _split_long_chunk(
                chunk,
                max_size=chunking_config.max_chunk_size_chars,
                overlap=chunking_config.chunk_overlap_chars,
            )
        )

    return final_chunks
