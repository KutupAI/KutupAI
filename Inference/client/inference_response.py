from dataclasses import dataclass
from typing import Optional
# الملف عبارة عن كلاس بيانات فقط (DTO)


@dataclass(slots=True)

# ينشئ Constructor تلقائياً.
#يحجز ذاكرة أقل.
#يمنع إضافة متغيرات غير معرفة.

class InferenceResponse:
    success: bool

    text: str

    model: Optional[str] = None

    prompt_tokens: int = 0

    completion_tokens: int = 0

    total_tokens: int = 0

    finish_reason: Optional[str] = None

    error: Optional[str] = None