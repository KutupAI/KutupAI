"""
retrieval_request.py
-----------------------
شكل موحّد لطلب الاسترجاع القادم من أي Agent (rag_agent, writer_agent).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RetrievalRequest:
    query: str                       # سؤال المستخدم أو استعلام الوكيل
    top_k: Optional[int] = None      # عدد النتائج - إن لم يُحدَّد يُستخدم الافتراضي
