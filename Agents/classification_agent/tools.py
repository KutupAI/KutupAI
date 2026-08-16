"""
tools.py
----------
External integrations needed by classification_agent:
- Optimization layer (fast ONNX pre-classification)
- Qwen VLM (multimodal classification via Inference/client/qwen_vl_client.py)

"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from Agents.classification_agent.config import ClassificationConfig
from Agents.classification_agent.exceptions import InvalidClassificationOutputError, QwenVLMError
from Agents.classification_agent.prompts import (
    SYSTEM_PROMPT,
    build_layout_summary,
    build_user_prompt,
)
from Agents.classification_agent.taxonomy import UNCERTAIN_CODE, is_valid_code
from Inference.client.qwen_vl_client import QwenVLClient, QwenVLRequest
from Optimization.services.fast_classification_service import (
    FastClassificationResult,
    classify_fast,
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def run_fast_classifier(normalized_text: str) -> FastClassificationResult | None:
    """Optimization-layer pre-filter. Returns None if unavailable/inconclusive
    -- see fast_classification_service.py for why that is a normal outcome.
    """
    return classify_fast(normalized_text)


def run_qwen_classification(
    *,
    normalized_text: str,
    ocr_confidence: float | None,
    layout: Any,
    image_bytes: bytes | None,
    config: ClassificationConfig,
) -> dict[str, Any]:
    """Call Qwen VLM and return a parsed+validated dict:
    {"document_type": str, "confidence": float, "alternatives": [...]}

    Raises QwenVLMError on transport failure, InvalidClassificationOutputError
    if the model's response isn't the strict JSON the prompt requires (§7).
    """
    layout_summary = build_layout_summary(layout) if config.use_layout_when_available else None

    user_prompt = build_user_prompt(
        normalized_text=normalized_text,
        ocr_confidence=ocr_confidence,
        layout_summary=layout_summary,
        top_k_alternatives=config.top_k_alternatives,
    )

    client = QwenVLClient(base_url=config.qwen_base_url, timeout=config.qwen_timeout_s)
    request = QwenVLRequest(
        text_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        image_bytes=image_bytes if config.send_image else None,
        temperature=config.qwen_temperature,
        max_tokens=config.qwen_max_tokens,
    )

    started = time.monotonic()
    response = client.generate(request)
    elapsed_ms = (time.monotonic() - started) * 1000

    if not response.success:
        raise QwenVLMError(response.error or "qwen_vl_client returned success=False")

    parsed = _parse_json_response(response.text)
    parsed["_processing_ms"] = elapsed_ms
    return parsed


def _parse_json_response(raw_text: str) -> dict[str, Any]:
    """Extract and validate the strict JSON the prompt demands.

    Model output can still occasionally wrap JSON in prose or a code fence
    despite instructions -- we take the first {...} block as a defensive
    measure, but do not silently invent fields that are missing.
    """
    match = _JSON_BLOCK_RE.search(raw_text or "")
    if not match:
        raise InvalidClassificationOutputError(f"No JSON object found in model output: {raw_text!r}")

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise InvalidClassificationOutputError(f"Model output was not valid JSON: {exc}") from exc

    document_type = data.get("document_type")
    if not isinstance(document_type, str) or not is_valid_code(document_type):
        # §5/§7: never invent a class outside the taxonomy. Degrade to the
        # explicit "uncertain" class rather than trust an out-of-taxonomy
        # label, and let the confidence stay whatever the model reported
        # (low confidence is expected here).
        data["document_type"] = UNCERTAIN_CODE

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)):
        raise InvalidClassificationOutputError(f"Missing/invalid 'confidence' in model output: {data!r}")
    data["confidence"] = max(0.0, min(1.0, float(confidence)))

    alternatives = data.get("alternatives") or []
    cleaned_alternatives = []
    for alt in alternatives:
        if not isinstance(alt, dict):
            continue
        alt_type = alt.get("type")
        alt_conf = alt.get("confidence")
        if isinstance(alt_type, str) and is_valid_code(alt_type) and isinstance(alt_conf, (int, float)):
            cleaned_alternatives.append({"type": alt_type, "confidence": max(0.0, min(1.0, float(alt_conf)))})
    data["alternatives"] = cleaned_alternatives

    return data
