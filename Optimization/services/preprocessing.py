"""
preprocessing.py
-------------------
Preprocessing helpers for the fast classification service.

Kept intentionally simple/deterministic (no ML) -- this runs before every
document, including ones that end up needing the full Qwen VLM path, so it
must stay cheap.
"""

from __future__ import annotations

import re


def clean_for_fast_classifier(text: str, *, max_chars: int = 2000) -> str:
    """Light normalization for the ONNX fast classifier's tokenizer.

    Not a replacement for ocr_agent's Turkish correction -- this only
    collapses whitespace and truncates, since the fast classifier is meant
    to be a cheap first pass, not a full NLP pipeline.
    """
    if not text:
        return ""
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed[:max_chars]
