"""
tools.py
----------
External integrations needed by classification_agent:
- Optimization layer (fast ONNX pre-classification)
- VLM (multimodal classification via Inference/client/vlm_client.py) --
  Gemma 3 (4B/12B/27B, local llama.cpp/llama-server), previously Qwen2.5-VL.
  See vlm_client.py's module docstring for the migration note.

Kept separate from agent.py so the control-flow/decision logic in agent.py
stays readable, matching the OCR agent's split between agent.py and
client.py/tools.py.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from Agents.classification_agent.config import ClassificationConfig
from Agents.classification_agent.exceptions import InvalidClassificationOutputError, VLMError
from Agents.classification_agent.prompts import (
    SYSTEM_PROMPT,
    build_layout_summary,
    build_user_prompt,
)
from Agents.classification_agent.taxonomy import UNCERTAIN_CODE, is_valid_code
from Inference.client.evren_client import EvrenClient
from Inference.client.inference_request import InferenceRequest, Message
from Inference.client.vlm_client import VLMClient, VLMRequest
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


def run_vlm_classification(
    *,
    normalized_text: str,
    ocr_confidence: float | None,
    layout: Any,
    ocr_pages: Any = None,
    image_bytes: bytes | None,
    config: ClassificationConfig,
) -> dict[str, Any]:
    """Call the classification VLM (Gemma 3, local llama.cpp/llama-server --
    see Inference/client/vlm_client.py) and return a parsed+validated dict:
    {"document_type": str, "confidence": float, "alternatives": [...]}

    Raises VLMError on transport failure, InvalidClassificationOutputError
    if the model's response isn't the strict JSON the prompt requires (§7).
    """
    from Agents.classification_agent.prompts import build_vision_signal_summary

    layout_summary = build_layout_summary(layout) if config.use_layout_when_available else None
    vision_summary = build_vision_signal_summary(ocr_pages) if config.use_layout_when_available else None

    user_prompt = build_user_prompt(
        normalized_text=normalized_text,
        ocr_confidence=ocr_confidence,
        layout_summary=layout_summary,
        vision_summary=vision_summary,
        top_k_alternatives=config.top_k_alternatives,
    )

    started = time.monotonic()
    if config.inference_backend == "evren":
        # API istemcisi aynı OpenAI mesaj biçimini kullanır; state sözleşmesi değişmez.
        response = EvrenClient(model=config.evren_model, timeout=config.vlm_timeout_s).generate(
            InferenceRequest(
                messages=[
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=user_prompt),
                ],
                temperature=config.vlm_temperature,
                max_tokens=config.vlm_max_tokens,
            )
        )
    else:
        client = VLMClient(base_url=config.vlm_base_url, timeout=config.vlm_timeout_s)
        response = client.generate(
            VLMRequest(
                text_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                image_bytes=image_bytes if config.send_image else None,
                model=config.vlm_model_name,
                temperature=config.vlm_temperature,
                max_tokens=config.vlm_max_tokens,
            )
        )
    elapsed_ms = (time.monotonic() - started) * 1000

    if not response.success:
        raise VLMError(response.error or "vlm_client returned success=False")

    parsed = _parse_json_response(response.text)
    parsed["_processing_ms"] = elapsed_ms
    return parsed


# Backward-compat alias: evaluation/ablation.py imports run_qwen_classification
# by name. Keeping this avoids touching that file in this change; new code
# should call run_vlm_classification directly.
run_qwen_classification = run_vlm_classification


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
