"""Signature / seal detection (visual + optional VL hints)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from Agents.ocr_agent.models import VisualElement


@dataclass(frozen=True)
class SignatureDetection:
    detected: bool
    handwritten: bool
    confidence: float | None
    bbox: list[float] | None


@dataclass(frozen=True)
class SealDetection:
    detected: bool
    confidence: float | None
    bbox: list[float] | None
    text: str | None


@dataclass(frozen=True)
class VisionHints:
    """Visual cues produced by PaddleOCR-VL (image inspected, not text keywords)."""

    signature_detected: bool = False
    signature_handwritten: bool = False
    stamp_detected: bool = False


class SignatureSealDetectorInterface(ABC):
    @abstractmethod
    def detect(
        self,
        page_text: str,
        visual_elements: list[VisualElement],
        image: np.ndarray | None = None,
        vision_hints: VisionHints | None = None,
    ) -> tuple[SignatureDetection, SealDetection]:
        raise NotImplementedError


class HeuristicSignatureSealDetector(SignatureSealDetectorInterface):
    """Layout visuals + ink/shape analysis. Text labels are not used as proof."""

    def detect(
        self,
        page_text: str,
        visual_elements: list[VisualElement],
        image: np.ndarray | None = None,
        vision_hints: VisionHints | None = None,
    ) -> tuple[SignatureDetection, SealDetection]:
        sig_visual = _best_match(visual_elements, ("signature", "handwrit", "handwriting"))
        seal_visual = _best_match(visual_elements, ("seal", "stamp", "mühür", "muhur"))

        sig_cv, hand_cv, sig_box, sig_conf = _detect_signature_ink(image)
        stamp_cv, stamp_box, stamp_conf = _detect_stamp(image)

        hint_sig = bool(vision_hints and vision_hints.signature_detected)
        hint_hand = bool(vision_hints and vision_hints.signature_handwritten)
        hint_stamp = bool(vision_hints and vision_hints.stamp_detected)

        sig_detected = bool(sig_visual is not None or sig_cv or hint_sig)
        handwritten = False
        if sig_detected:
            handwritten = bool(
                (sig_visual is not None and "hand" in (sig_visual.element_type or "").lower())
                or hand_cv
                or hint_hand
                or sig_cv
            )

        signature = SignatureDetection(
            detected=sig_detected,
            handwritten=handwritten,
            confidence=(
                sig_visual.confidence if sig_visual is not None
                else (sig_conf if sig_cv else (0.75 if hint_sig else None))
            ),
            bbox=(
                sig_visual.bounding_box.as_xyxy() if sig_visual is not None
                else (sig_box if sig_cv else None)
            ),
        )

        stamp_detected = bool(seal_visual is not None or stamp_cv or hint_stamp)
        seal = SealDetection(
            detected=stamp_detected,
            confidence=(
                seal_visual.confidence if seal_visual is not None
                else (stamp_conf if stamp_cv else (0.75 if hint_stamp else None))
            ),
            bbox=(
                seal_visual.bounding_box.as_xyxy() if seal_visual is not None
                else (stamp_box if stamp_cv else None)
            ),
            text=None,
        )
        return signature, seal


def _best_match(elements: list[VisualElement], keywords: tuple[str, ...]) -> VisualElement | None:
    candidates = [
        ve for ve in elements if any(k in ve.element_type.lower() for k in keywords)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda ve: ve.confidence)


def _detect_signature_ink(
    image: np.ndarray | None,
) -> tuple[bool, bool, list[float] | None, float | None]:
    if image is None or getattr(image, "size", 0) == 0:
        return False, False, None, None

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Blue / cyan ballpoint (often desaturated in phone photos).
    blue = cv2.inRange(hsv, (80, 18, 30), (145, 255, 255))
    green = cv2.inRange(hsv, (40, 25, 30), (79, 255, 255))
    ink = cv2.bitwise_or(blue, green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = float(h * w)
    best = None
    best_score = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        ratio = area / page_area
        if ratio < 0.00025 or ratio > 0.06:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        # Header logos / brand marks are not signatures.
        if x <= 3 or y <= 3 or (x + bw) >= (w - 3) or (y + bh) >= (h - 3):
            continue  # photo border / background, not page ink
        # CamScanner / app watermarks sit on the footer corners.
        if y > 0.80 * h and (x < 0.18 * w or x > 0.62 * w):
            continue
        if y < 0.16 * h:
            continue
        if y < 0.22 * h and x < 0.28 * w and bw < 0.3 * w:
            continue
        # Skip QR/logo squares.
        aspect = bw / max(1.0, float(bh))
        if 0.75 <= aspect <= 1.35 and ratio < 0.012 and y < 0.18 * h:
            continue
        peri = cv2.arcLength(contour, True)
        circularity = 4.0 * np.pi * area / (peri * peri + 1e-6)
        if circularity > 0.75:
            continue  # round stamp-like blob, handled by stamp detector
        complexity = peri * peri / (area + 1e-6)
        if aspect < 1.15 and complexity < 25:
            continue
        score = min(0.95, 0.55 + min(0.3, ratio * 40) + min(0.15, complexity / 200.0))
        if score > best_score:
            best_score = score
            best = [float(x), float(y), float(x + bw), float(y + bh)]

    if best is None:
        return False, False, None, None
    return True, True, best, round(best_score, 4)


def _detect_stamp(
    image: np.ndarray | None,
) -> tuple[bool, list[float] | None, float | None]:
    if image is None or getattr(image, "size", 0) == 0:
        return False, None, None

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 60, 40), (12, 255, 255))
    red2 = cv2.inRange(hsv, (168, 60, 40), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)
    # Official stamps are often red; some are blue/navy circular seals.
    blue = cv2.inRange(hsv, (90, 70, 40), (140, 255, 255))
    color = cv2.bitwise_or(red, blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, kernel, iterations=2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    min_r = max(14, int(0.025 * min(h, w)))
    max_r = max(min_r + 4, int(0.12 * min(h, w)))
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(0.12 * min(h, w)),
        param1=80,
        param2=28,
        minRadius=min_r,
        maxRadius=max_r,
    )

    page_area = float(h * w)
    best = None
    best_score = 0.0

    if circles is not None:
        for (cx, cy, r) in np.round(circles[0]).astype(int):
            # Skip CamScanner-style watermarks at the page corners.
            if cy > 0.82 * h and (cx < 0.18 * w or cx > 0.82 * w):
                continue
            if cy < 0.12 * h and cx < 0.18 * w:
                continue  # top-left logo
            x0 = max(0, cx - r)
            y0 = max(0, cy - r)
            x1 = min(w, cx + r)
            y1 = min(h, cy + r)
            roi = color[y0:y1, x0:x1]
            color_ratio = float(np.mean(roi > 0)) if roi.size else 0.0
            if color_ratio < 0.08:
                continue  # gray/black app watermark or empty circle
            score = min(0.95, 0.55 + color_ratio)
            if score > best_score:
                best_score = score
                best = [float(x0), float(y0), float(x1), float(y1)]

    contours, _ = cv2.findContours(color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        ratio = area / page_area
        if ratio < 0.001 or ratio > 0.08:
            continue
        peri = cv2.arcLength(contour, True)
        circularity = 4.0 * np.pi * area / (peri * peri + 1e-6)
        if circularity < 0.65:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if y > 0.82 * h and (x < 0.12 * w or x + bw > 0.88 * w):
            continue
        score = min(0.93, 0.5 + circularity * 0.4)
        if score > best_score:
            best_score = score
            best = [float(x), float(y), float(x + bw), float(y + bh)]

    if best is None:
        return False, None, None
    return True, best, round(best_score, 4)


def build_detector() -> SignatureSealDetectorInterface:
    return HeuristicSignatureSealDetector()
