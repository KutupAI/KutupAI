"""
Paylaşılan Türkçe/Hukuki Metin Tokenizasyon Yardımcıları
--------------------------------------------------------
BM25 ve PRF için metinleri anlamlı parçalara (token) ayırır.
Hukuki bağlamda kritik olan kelimeler (madde, kanun vb.) Stopword olarak atılmaz.
"""

from __future__ import annotations

import re
from typing import List

# Türkçe karakterleri ve rakamları yakala.
_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿĞğİıŞşÇçÜüÖö]+", re.UNICODE)
_TURKISH_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")

# Hukukî aramada önemli sözcükleri durdurma listesine ekleme.
STOPWORDS = {
    "ve", "ile", "bir", "bu", "için", "icin", "olan", "olarak", "veya", "de", "da",
    "ki", "mi", "mı", "mu", "mü", "the", "and", "or", "of", "to", "in", "on",
    "göre", "gore", "her", "gibi", "olarak", "üzere", "uzere", "veya", "yahut",
}


def fold_turkish(text: str) -> str:
    """Fold Turkish characters so accented and ASCII keyboard queries match."""
    return (text or "").translate(_TURKISH_ASCII).replace("\u0307", "")


def tokenize(text: str, *, min_len: int = 1, drop_stopwords: bool = False) -> List[str]:
    """Metni küçük harfe çevirip tokenlere ayırır."""
    tokens = [fold_turkish(t.lower()) for t in _TOKEN_RE.findall(text or "")]
    tokens = [t for t in tokens if len(t) >= min_len]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
