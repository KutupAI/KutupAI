"""Hardware-agnostic device resolution (auto → gpu:0 or cpu)."""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def detect_gpu() -> bool:
    """True if Paddle or Torch reports a CUDA device."""
    try:
        import paddle  # type: ignore

        if bool(paddle.device.cuda.device_count()):
            return True
    except Exception:
        pass

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return True
    except Exception:
        pass

    return False


def resolve_device(requested: str) -> str:
    """Map auto/cpu/gpu/cuda[:N] to a PaddleOCR device string.

    Default preference is GPU (ocr_agent runs on GPU when CUDA is present).
    """
    value = (requested or "gpu").strip().lower()

    if value in {"cpu"}:
        return "cpu"

    if value in {"auto", ""}:
        resolved = "gpu:0" if detect_gpu() else "cpu"
        logger.info("OCR device auto-resolved to %s", resolved)
        return resolved

    if value.startswith("cuda"):
        value = "gpu" + value[len("cuda"):]

    if value in {"gpu", "gpu:0"} or value.startswith("gpu:"):
        if not detect_gpu():
            logger.warning(
                "Device '%s' requested but no GPU detected; using CPU.",
                requested,
            )
            return "cpu"
        resolved = "gpu:0" if value == "gpu" else value
        logger.info("OCR device resolved to %s", resolved)
        return resolved

    logger.warning("Unrecognized OCR device '%s'; defaulting to CPU.", requested)
    return "cpu"
