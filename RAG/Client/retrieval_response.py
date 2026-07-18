"""
retrieval_response.py
------------------------
شكل موحّد لاستجابة الاسترجاع المُعادة إلى الـ Agent المستدعي.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class RetrievalResponse:
    context: str                          # نص جاهز مُنسّق (من context_formatter)
    sources: List[Dict[str, Any]] = field(default_factory=list)  # مصادر مختصرة
    result_count: int = 0                 # عدد المقاطع الفعلية المسترجعة
