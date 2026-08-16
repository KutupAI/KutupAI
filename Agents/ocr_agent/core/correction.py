from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CorrectionDecision:
    text: str
    applied: bool


class TurkishOCRCorrector:
    """Conservative OCR correction for Turkish administrative documents.

    It intentionally avoids generative rewriting. Corrections are limited to
    deterministic OCR artifacts that can be validated locally.
    """

    _spaces = re.compile(r"[ \t]+")
    _space_before_punct = re.compile(r"\s+([,.;:!?])")
    _broken_numeric = re.compile(r"(?<=\d)\s+(?=\d)")
    _turkish_quotes = {
        "“": '"', "”": '"', "‘": "'", "’": "'",
    }

    def correct(self, text: str) -> CorrectionDecision:
        if not text:
            return CorrectionDecision(text="", applied=False)

        original = text
        for src, dst in self._turkish_quotes.items():
            text = text.replace(src, dst)

        text = self._spaces.sub(" ", text)
        text = self._space_before_punct.sub(r"\1", text)
        text = self._broken_numeric.sub("", text)
        text = text.strip()

        return CorrectionDecision(text=text, applied=(text != original))
