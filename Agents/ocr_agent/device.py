"""Hardware-agnostic device resolution.

`OCRConfig.device` is "auto" by default. This module is the ONLY place that
inspects the actual machine (GPU presence, driver) and turns "auto" into a
concrete device string. Nothing else in the OCR Agent should hard-code a
CUDA device id, GPU model, or CPU model — that keeps the Agent portable
between the RTX 3050 dev box and a future production server.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def detect_gpu() -> bool:
    """Best-effort GPU availability check. Cached: probed once per process."""
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
    """Turn a configured device string into what PaddleOCR/PP-StructureV3 expects.

    Accepts: "auto", "cpu", "gpu", "gpu:N", "cuda", "cuda:N".
    """
    value = (requested or "auto").strip().lower()

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
                "Device '%s' was requested but no GPU was detected; falling back to CPU.",
                requested,
            )
            return "cpu"
        return value

    # Unknown value: don't guess silently, but don't crash the pipeline either.
    logger.warning("Unrecognized OCR device '%s'; defaulting to CPU.", requested)
    return "cpu"
