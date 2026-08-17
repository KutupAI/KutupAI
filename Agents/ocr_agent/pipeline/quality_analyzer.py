"""Cheap, deterministic image-quality scoring.

Used to decide *whether* a page needs extra preprocessing/retries before
paying for another OCR pass — per requirement #4 ("adaptive", not
"aggressively preprocess every page").
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityScore:
    score: float  # 0..1, higher is better
    blur: float
    brightness: float
    contrast: float
    readable: bool

    @property
    def is_dark(self) -> bool:
        return self.brightness < 0.38

    @property
    def is_blurry(self) -> bool:
        return self.blur < 0.28

    @property
    def is_low_contrast(self) -> bool:
        return self.contrast < 0.22

    @property
    def poor(self) -> bool:
        return (not self.readable) or self.is_dark or self.is_blurry or self.is_low_contrast


def assess(image: np.ndarray, quality_threshold: float) -> QualityScore:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Sharpness via variance of Laplacian (higher = sharper).
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = min(1.0, laplacian_var / 250.0)

    brightness = float(np.mean(gray)) / 255.0
    # Penalize both too-dark and blown-out images; 0.5 is ideal midtone.
    brightness_score = 1.0 - min(1.0, abs(brightness - 0.5) * 2.2)

    contrast = float(np.std(gray)) / 128.0
    contrast_score = min(1.0, contrast)

    score = max(0.0, min(1.0, 0.5 * blur_score + 0.25 * brightness_score + 0.25 * contrast_score))

    return QualityScore(
        score=score,
        blur=blur_score,
        brightness=brightness,
        contrast=contrast_score,
        readable=score >= quality_threshold,
    )
