"""Shared Turkish/legal text tokenization helpers."""

from __future__ import annotations

import re
from typing import List

_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿĞğİıŞşÇçÜüÖö]+", re.UNICODE)

STOPWORDS = {
    "ve", "ile", "bir", "bu", "için", "icin", "olan", "olarak", "veya", "de", "da",
    "ki", "mi", "mı", "mu", "mü", "the", "and", "or", "of", "to", "in", "on",
    "madde", "kanun", "sayılı", "sayili", "göre", "gore", "her", "gibi",
}


def tokenize(text: str, *, min_len: int = 1, drop_stopwords: bool = False) -> List[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    tokens = [t for t in tokens if len(t) >= min_len]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
