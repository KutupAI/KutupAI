"""
classification_agent -- decides the document type for Orchestration.

Pipeline (Documentation/architecture.md section 4 / task doc section 4+7):
  1. Take OCR output already in graph_state (produced by ocr_agent -- this
     Agent does not run OCR itself).
  2. Try the Optimization fast classifier first (cheap ONNX pre-filter).
  3. If unavailable, inconclusive, or below the escalation threshold,
     call the VLM (Gemma 3, local llama.cpp/llama-server) with
     normalized_text + rendered image + layout together.
  4. Apply the needs_review threshold and return strict JSON.

Unified pipeline envelope contract (read / write):

  Input  (classification empty):
    {request, ocr, classification: {}, extraction, validation,
     rag, summary, routing, writing}

  Output (same envelope, classification filled):
    classification: {
      "success": bool,
      "document_type": str,
      "classification_confidence": float
    }

Also writes Orchestration wire keys classification_result / classification_status
for GraphState compatibility. Never calls Storage; never runs OCR itself.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Dict, Optional

from Agents.base.agent_registry import register
from Agents.base.base_agent import BaseAgent
from Agents.classification_agent.config import ClassificationConfig
from Agents.classification_agent.exceptions import ClassificationAgentError, MissingInputError
from Agents.classification_agent.models import ClassificationAlternative, ClassificationResult
from Agents.classification_agent.taxonomy import UNCERTAIN_CODE
from Agents.classification_agent.tools import run_fast_classifier, run_vlm_classification

# Canonical keys written to state["classification"] (unified contract).
CLASSIFICATION_CONTRACT_KEYS = ("success", "document_type", "classification_confidence")


@register
class ClassificationAgent(BaseAgent):
    """Classify document type only -- no extraction, validation, RAG, or
    Storage writes. Always returns strict JSON (never free-text)."""

    name = "classification_agent"
    description = "Classify a document into one of the 18 task-defined classes using OCR text + image + layout."

    def __init__(self, config: ClassificationConfig | None = None) -> None:
        self.config = config or ClassificationConfig.from_env()
        self.config.validate()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestration / envelope entry point.

        Accepts either the unified pipeline envelope (``request`` / ``ocr``)
        or legacy GraphState wire keys (``ocr_result`` / ``document_text``).
        Writes only ``classification`` (+ ``classification_result`` /
        ``classification_status`` mirrors); all other envelope keys pass through.
        """
        if not isinstance(state, dict):
            raise TypeError("ClassificationAgent.run expects GraphState / envelope as a dict")

        updated = dict(state)
        started = time.monotonic()
        document_id = _resolve_document_id(state)

        try:
            normalized_text, ocr_confidence, layout, ocr_pages, image_bytes, document_id = _extract_inputs(state)
        except MissingInputError as exc:
            result = ClassificationResult(
                success=False,
                document_id=document_id,
                document_type=None,
                confidence=0.0,
                status="failed",
                source="none",
                error=str(exc),
                processing_ms=(time.monotonic() - started) * 1000,
            )
            return _merge_result(updated, result)

        try:
            result = self._classify(
                document_id=document_id,
                normalized_text=normalized_text,
                ocr_confidence=ocr_confidence,
                layout=layout,
                ocr_pages=ocr_pages,
                image_bytes=image_bytes,
            )
        except ClassificationAgentError as exc:
            result = ClassificationResult(
                success=False,
                document_id=document_id,
                document_type=None,
                confidence=0.0,
                status="failed",
                source="error",
                ocr_confidence=ocr_confidence,
                error=str(exc),
            )

        result.processing_ms = (time.monotonic() - started) * 1000
        return _merge_result(updated, result)

    def process(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """Standalone envelope entry (same contract as ``run``).

        Input : {request, ocr, classification: {}, …}
        Output: same envelope with ``classification`` filled per contract.
        """
        return self.run(envelope)

    def _classify(
        self,
        *,
        document_id: str | None,
        normalized_text: str,
        ocr_confidence: float | None,
        layout: Any,
        ocr_pages: Any,
        image_bytes: bytes | None,
    ) -> ClassificationResult:
        # Step 1: fast path (Optimization / ONNX)
        if self.config.use_fast_classifier:
            fast = run_fast_classifier(normalized_text)
            if fast is not None and fast.confidence >= self.config.fast_classifier_escalation_threshold:
                return self._build_result(
                    document_id=document_id,
                    document_type=fast.document_type,
                    confidence=fast.confidence,
                    alternatives=[{"type": t, "confidence": c} for t, c in fast.alternatives],
                    ocr_confidence=ocr_confidence,
                    source="optimization_fast",
                )
            # else: fast classifier absent or not confident enough -> escalate to VLM.

        # Step 2: VLM (section 4: image + text + layout together). Gemma 3
        # (4B/12B/27B, local llama.cpp/llama-server) -- see Inference/client/vlm_client.py.
        vlm_output = run_vlm_classification(
            normalized_text=normalized_text,
            ocr_confidence=ocr_confidence,
            layout=layout,
            ocr_pages=ocr_pages,
            image_bytes=image_bytes,
            config=self.config,
        )
        return self._build_result(
            document_id=document_id,
            document_type=vlm_output["document_type"],
            confidence=vlm_output["confidence"],
            alternatives=vlm_output.get("alternatives", []),
            ocr_confidence=ocr_confidence,
            source="vlm",
        )

    def _build_result(
        self,
        *,
        document_id: str | None,
        document_type: str,
        confidence: float,
        alternatives: list[dict[str, Any]],
        ocr_confidence: float | None,
        source: str,
    ) -> ClassificationResult:
        # Section 7: low confidence -> needs_review, but still report the
        # best-guess document_type (never force-assign a class the model
        # was NOT confident about, and never silently discard the guess).
        status = "success" if confidence >= self.config.needs_review_threshold else "needs_review"
        alts = alternatives[: self.config.top_k_alternatives]

        return ClassificationResult(
            success=True,
            document_id=document_id,
            document_type=document_type or UNCERTAIN_CODE,
            confidence=confidence,
            alternatives=[ClassificationAlternative(**a) for a in alts],
            status=status,
            source=source,
            ocr_confidence=ocr_confidence,
        )


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _resolve_document_id(state: Dict[str, Any]) -> Optional[str]:
    request = _as_dict(state.get("request"))
    document = _as_dict(request.get("document"))
    return (
        document.get("document_id")
        or state.get("document_id")
        or request.get("document_id")
        or None
    )


def _resolve_ocr_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer unified ``state["ocr"]["ocr_data"]``, then legacy ``ocr_result``."""
    ocr = _as_dict(state.get("ocr"))
    ocr_data = _as_dict(ocr.get("ocr_data"))
    if ocr_data.get("full_text") is not None or ocr_data.get("pages") is not None:
        # Promote document-level vision onto pages when pages are empty so
        # prompts.py can still consume signature/stamp signals.
        pages = list(ocr_data.get("pages") or [])
        vision = ocr_data.get("vision")
        if not pages and isinstance(vision, dict):
            pages = [{"page_number": 1, "text": ocr_data.get("full_text") or "", "vision": vision}]
        return {
            "full_text": ocr_data.get("full_text") or "",
            "pages": pages,
            "page_count": ocr_data.get("page_count"),
            "language": ocr_data.get("language"),
            "vision": vision,
            "success": ocr.get("success"),
        }

    ocr_result = _as_dict(state.get("ocr_result"))
    # Wire envelope: { Success, Data: [document, ...] }
    data = ocr_result.get("Data") or ocr_result.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        doc = data[0]
        return {
            "full_text": doc.get("full_text") or "",
            "pages": list(doc.get("pages") or []),
            "page_count": doc.get("page_count"),
            "language": doc.get("language"),
            "vision": doc.get("vision"),
            "document_id": doc.get("document_id"),
            "success": ocr_result.get("Success", ocr_result.get("success")),
        }

    if "full_text" in ocr_result or "pages" in ocr_result:
        return {
            "full_text": ocr_result.get("full_text") or "",
            "pages": list(ocr_result.get("pages") or []),
            "page_count": ocr_result.get("page_count"),
            "language": ocr_result.get("language"),
            "vision": ocr_result.get("vision"),
            "document_id": ocr_result.get("document_id"),
            "success": ocr_result.get("success"),
        }

    return {}


def _extract_inputs(
    state: Dict[str, Any],
) -> tuple[str, float | None, Any, Any, bytes | None, str | None]:
    """Pull classification inputs from the unified envelope or legacy GraphState.

    Raises MissingInputError only when there is truly nothing to classify
    from (no text AND no image).
    """
    ocr_payload = _resolve_ocr_payload(state)
    ocr_pages = ocr_payload.get("pages") or []

    normalized_text = (
        state.get("normalized_text")
        or ocr_payload.get("full_text")
        or state.get("document_text")
        or state.get("text")
        or ""
    )
    ocr_confidence = _extract_ocr_confidence(ocr_payload)
    layout = [p.get("layout") for p in ocr_pages if isinstance(p, dict) and p.get("layout")] or None
    document_id = _resolve_document_id(state) or ocr_payload.get("document_id")

    image_bytes = _extract_image_bytes(state)

    if not str(normalized_text).strip() and image_bytes is None:
        raise MissingInputError(
            "classification_agent: no OCR text and no document image available in state"
        )

    return str(normalized_text), ocr_confidence, layout, ocr_pages, image_bytes, document_id


def _extract_ocr_confidence(ocr_payload: Dict[str, Any]) -> float | None:
    """Average per-item confidence when present; otherwise None."""
    text_items = []
    for page in ocr_payload.get("pages") or []:
        if not isinstance(page, dict):
            continue
        text_items.extend(page.get("text_items") or [])
    confidences = [
        ti.get("confidence")
        for ti in text_items
        if isinstance(ti, dict) and ti.get("confidence") is not None
    ]
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def _extract_image_bytes(state: Dict[str, Any]) -> bytes | None:
    """Best-effort: state may carry a pre-rendered page image as bytes/base64."""
    image = state.get("document_image") or state.get("page_image")
    if image is None:
        return None
    if isinstance(image, bytes):
        return image
    if isinstance(image, str):
        try:
            return base64.b64decode(image)
        except Exception:
            return None
    return None


def _classification_contract(result: ClassificationResult) -> Dict[str, Any]:
    """Exact unified-contract shape for state['classification']."""
    return {
        "success": bool(result.success),
        "document_type": result.document_type,
        "classification_confidence": round(float(result.confidence or 0.0), 4),
    }


def _merge_result(state: Dict[str, Any], result: ClassificationResult) -> Dict[str, Any]:
    # Dual-key convention (same as validation_agent): unified short key +
    # Orchestration wire key carry the same contract payload so StateManager
    # can keep either without losing the envelope shape.
    contract = _classification_contract(result)
    state["classification"] = contract
    state["classification_result"] = contract
    state["classification_status"] = result.status
    # Unified short-key contract for validation_agent, matching ocr_agent's
    # existing dual-key convention (state["ocr"] short/unified key +
    # state["ocr_result"] wire key). classification_agent previously only
    # wrote "classification_result" (with a field named "confidence"), so
    # validation_agent's state.get("classification") / "classification_confidence"
    # lookup always saw an empty dict, silently skipping the
    # low_classification_confidence check on every run.
    state["classification"] = {
        "success": result.success,
        "classification_confidence": result.confidence,
        "document_type": result.document_type,
        "status": result.status,
    }
    if not result.success:
        errors = list(state.get("errors") or [])
        errors.append(f"classification_agent: {result.error}")
        state["errors"] = errors
    return state


def process(envelope: Dict[str, Any], agent: Optional[ClassificationAgent] = None) -> Dict[str, Any]:
    """Module-level envelope entry matching routing_agent.process style."""
    return (agent or ClassificationAgent()).process(envelope)
