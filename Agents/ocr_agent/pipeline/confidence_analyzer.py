"""Decide what to do after an OCR attempt: accept, retry, or escalate.

Implements the controlled multi-pass strategy from requirement #5:

    Attempt 1 -> confidence check
    Good      -> accept
    Poor      -> preprocessing -> Attempt 2
    Still poor-> stronger preprocessing -> Attempt 3
    Still bad -> vision fallback (if enabled) -> mark uncertain otherwise
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import cv2
import numpy as np

from Agents.ocr_agent.core.ocr_parser import join_page_text
from Agents.ocr_agent.models import OCRTextItem
from Agents.ocr_agent.pipeline.quality_analyzer import QualityScore


class NextAction(str, Enum):
    ACCEPT = "accept"
    RETRY = "retry"
    FALLBACK = "fallback"
    GIVE_UP = "give_up"


@dataclass(frozen=True)
class ConfidenceDecision:
    action: NextAction
    mean_confidence: float
    low_confidence_ratio: float
    incomplete: bool = False
    quality_poor: bool = False
    corrupted: bool = False


def page_confidence(items: Sequence[OCRTextItem]) -> float:
    if not items:
        return 0.0
    return sum(i.confidence for i in items) / len(items)


_TR_LETTERS = set("çğıöşüÇĞİÖŞÜ")
_ASCII_TR_HINT = re.compile(
    r"(?i)\b(?:t[uü]ketici|s[iıi]rket|s[oö]zle[sş]me|y[oö]netmelik|"
    r"bildirim|uyusmazl[iı]k|mahkemelere?|hukuk|mevzuat|h[uü]k[uü]m)\b"
)
_MIDLINE_CLAUSE = re.compile(r"\S{10,}\s+\(\d+\)\s+\S{10,}")
_GARBAGE_CAMEL = re.compile(r"^[A-Za-zÇĞİÖŞÜçğıöşü]*[A-ZÇĞİÖŞÜ][a-zçğıöşü]+[A-ZÇĞİÖŞÜ]")


def page_looks_incomplete(
    image: np.ndarray | None,
    items: Sequence[OCRTextItem],
    quality: QualityScore | None = None,
) -> bool:
    """Detect truncated / obviously missing OCR without calling Qwen yet."""

    if not items:
        return True

    text = join_page_text(items)
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    density = _edge_density(image) if image is not None else 0.0

    # A short result is incomplete only when the image itself looks like a
    # dense document (many text edges). Blank/unit-test pages stay complete.
    if len(words) < 8:
        return density >= 0.08

    if image is None:
        return False

    density = _edge_density(image)
    edge_pixels = density * float(image.shape[0] * image.shape[1])
    expected_words = edge_pixels / 450.0
    if expected_words > 60 and len(words) < expected_words * 0.55:
        return True

    if _missing_text_band(image, items):
        return True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    alpha_lines = [ln for ln in lines if any(c.isalpha() for c in ln)]
    if len(alpha_lines) >= 8:
        fragments = sum(1 for ln in alpha_lines if len(ln) <= 8)
        if fragments / len(alpha_lines) >= 0.35:
            return True

    if quality is not None and quality.poor and len(words) < 40:
        return True

    return False


def page_looks_corrupted(items: Sequence[OCRTextItem]) -> bool:
    """Suspicious merges, missing Turkish letters, or garbage tokens."""

    if not items:
        return False
    text = join_page_text(items)
    if not text.strip():
        return False

    letters = [c for c in text if c.isalpha()]
    tr_hits = sum(1 for c in letters if c in _TR_LETTERS) if letters else 0
    ascii_hints = _ASCII_TR_HINT.findall(text)
    if len(ascii_hints) >= 3 and tr_hits < 3:
        return True

    mixed = sum(1 for ln in text.splitlines() if _MIDLINE_CLAUSE.search(ln or ""))
    if mixed >= 2:
        return True

    garbage = 0
    for token in re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]{4,}", text):
        if _GARBAGE_CAMEL.match(token):
            garbage += 1
            continue
        vowels = len(re.findall(r"[aeiouAEIOUıiöüİ]", token))
        if len(token) >= 6 and vowels == 0:
            garbage += 1
    if garbage >= 3:
        return True

    return reading_order_unreliable(items)


def reading_order_unreliable(items: Sequence[OCRTextItem]) -> bool:
    """Detect interleaved columns / stacked boxes joined as one line."""

    text = join_page_text(items)
    clause_at_start = 0
    clause_mid = 0
    for ln in text.splitlines():
        stripped = ln.strip()
        if re.match(r"^\(\d+\)", stripped):
            clause_at_start += 1
        elif re.search(r"\S.+\(\d+\)", stripped):
            clause_mid += 1
    if clause_mid >= 2 and clause_mid >= clause_at_start:
        return True

    article_glued = len(re.findall(r"(?i)Madde\s*\d+\s*-\s*\S{8,}", text))
    spaced = len(re.findall(r"(?i)Madde\s+\d+\s*-\s+\S", text))
    if article_glued >= 2 and spaced == 0 and "," not in text[:80]:
        # Not by itself enough; only when glued headings also collide with body.
        if re.search(r"(?i)Madde\s*\d+\S+\s+\(\d+\)", text):
            return True
    return False


def page_needs_vision(
    image: np.ndarray | None,
    items: Sequence[OCRTextItem],
    quality: QualityScore | None = None,
) -> tuple[bool, str]:
    if page_looks_incomplete(image, items, quality):
        return True, "incomplete"
    if page_looks_corrupted(items):
        return True, "corrupted"
    if reading_order_unreliable(items):
        return True, "reading_order"
    return False, ""


def decide(
    items: Sequence[OCRTextItem],
    *,
    attempt: int,
    max_attempts: int,
    low_confidence_threshold: float,
    fallback_enabled: bool,
    fallback_threshold: float,
    incomplete: bool = False,
    quality_poor: bool = False,
    corrupted: bool = False,
) -> ConfidenceDecision:
    mean_conf = page_confidence(items)

    if not items:
        low_ratio = 1.0
    else:
        low_ratio = sum(1 for i in items if i.confidence < low_confidence_threshold) / len(items)

    coverage_bad = incomplete or corrupted
    needs_help = coverage_bad or (not items) or mean_conf < low_confidence_threshold or low_ratio >= 0.3

    # Good enough: accept immediately, no wasted retries/fallback.
    # Dark/blurry pages still accept if OCR itself is complete and confident.
    if (
        items
        and mean_conf >= low_confidence_threshold
        and low_ratio < 0.3
        and not coverage_bad
    ):
        return ConfidenceDecision(
            NextAction.ACCEPT, mean_conf, low_ratio, incomplete, quality_poor, corrupted
        )

    # Coverage / corruption will not be fixed by another RapidOCR pass when
    # the current mean confidence is already high — go to Qwen.
    if coverage_bad and fallback_enabled and mean_conf >= low_confidence_threshold:
        return ConfidenceDecision(
            NextAction.FALLBACK, mean_conf, low_ratio, incomplete, quality_poor, corrupted
        )

    # Still budget left for another (stronger) OCR attempt.
    if attempt < max_attempts:
        return ConfidenceDecision(
            NextAction.RETRY, mean_conf, low_ratio, incomplete, quality_poor, corrupted
        )

    # Out of OCR attempts: escalate to vision fallback only if truly needed.
    if fallback_enabled and (
        mean_conf < fallback_threshold
        or coverage_bad
        or quality_poor
        or not items
        or needs_help
    ):
        return ConfidenceDecision(
            NextAction.FALLBACK, mean_conf, low_ratio, incomplete, quality_poor, corrupted
        )

    return ConfidenceDecision(
        NextAction.GIVE_UP, mean_conf, low_ratio, incomplete, quality_poor, corrupted
    )


def _edge_density(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 40, 120)
    return float(np.mean(edges > 0))


def _missing_text_band(image: np.ndarray, items: Sequence[OCRTextItem]) -> bool:
    h, w = image.shape[:2]
    if h < 80 or w < 80:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 40, 120)
    bands = 8
    active = 0
    missing = 0
    for b in range(bands):
        y0 = int(h * b / bands)
        y1 = int(h * (b + 1) / bands)
        energy = float(np.mean(edges[y0:y1] > 0))
        if energy < 0.030:
            continue
        active += 1
        overlap = 0
        for item in items:
            _, iy0, _, iy1 = item.bounding_box.as_xyxy()
            if iy1 < y0 or iy0 > y1:
                continue
            overlap += 1
        if overlap < 1:
            missing += 1
    if active >= 3 and missing >= 1:
        return True

    # Tight top/bottom strips: first/last visible lines are often clipped.
    for y0, y1 in ((0, int(0.08 * h)), (int(0.92 * h), h)):
        if y1 <= y0:
            continue
        energy = float(np.mean(edges[y0:y1] > 0))
        if energy < 0.040:
            continue
        covered = any(
            not (item.bounding_box.as_xyxy()[3] < y0 or item.bounding_box.as_xyxy()[1] > y1)
            for item in items
        )
        if not covered:
            return True
    return False
