"""
classification_agent -- decides the document type for Orchestration.

Pipeline (Documentation/architecture.md section 4 / task doc section 4+7):
  1. Take OCR output already in graph_state (produced by ocr_agent -- this
     Agent does not run OCR itself).
  2. Try the Optimization fast classifier first (cheap ONNX pre-filter).
  3. If unavailable, inconclusive, or below the escalation threshold,
     call the VLM (Gemma 3, local llama.cpp/llama-server) with
     normalized_text + rendered image + layout together.
  4. Apply the needs_review threshold and return strict JSON in
     graph_state["classification_result"] (never free text -- section 7).

classification_agent never calls Storage directly (base_agent.py contract)
and never runs OCR itself -- it only consumes ocr_result from state,
matching the Supervisor's OCR -> Classification routing rule
(Orchestration/supervisor/routing_logic.py).
"""

from __future__ import annotations

import base64
import time
from typing import Any, Dict

from Agents.base.agent_registry import register
from Agents.base.base_agent import BaseAgent
from Agents.classification_agent.config import ClassificationConfig
from Agents.classification_agent.exceptions import ClassificationAgentError, MissingInputError
from Agents.classification_agent.models import ClassificationAlternative, ClassificationResult
from Agents.classification_agent.taxonomy import UNCERTAIN_CODE
from Agents.classification_agent.tools import run_fast_classifier, run_vlm_classification


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
        updated = dict(state)
        started = time.monotonic()

        try:
            normalized_text, ocr_confidence, layout, ocr_pages, image_bytes, document_id = _extract_inputs(state)
        except MissingInputError as exc:
            result = ClassificationResult(
                success=False,
                document_id=state.get("document_id"),
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
            # else: fast classifier absent or not confident enough -> escalate to Qwen VLM.

        # Step 2: VLM (section 4: image + text + layout together). Gemma 3
        # (4B/12B/27B, local llama.cpp/llama-server) as of this change --
        # previously Qwen2.5-VL, see Inference/client/vlm_client.py.
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


def _extract_inputs(
    state: Dict[str, Any],
) -> tuple[str, float | None, Any, Any, bytes | None, str | None]:
    """Pull classification_agent's inputs out of graph_state, produced
    upstream by ocr_agent. Raises MissingInputError only when there is
    truly nothing to classify from (no text AND no image).

    IMPORTANT: `state["ocr_result"]` is expected to be the `data` object
    from ocr_agent's response (i.e. `ocr_response["data"]`), not the full
    wrapper with `success`/`status` -- confirmed against real ocr_agent
    output samples. Real shape:
        {"document_id", "file_name", "file_type", "page_count", "language",
         "pages": [{"page_number", "text", "vision": {"signature": {...}, "stamp": {...}}}],
         "full_text"}
    Note ocr_agent does not emit a `text_items`/`confidence` or `layout`
    key per page -- earlier versions of this function assumed those and
    would have silently always gotten ocr_confidence=None / no layout.
    `vision` (signature/stamp detection) is real signal and is threaded
    through separately as `ocr_pages` for prompts.py to use.
    """
    ocr_result = state.get("ocr_result") or {}
    ocr_pages = ocr_result.get("pages") or []

    normalized_text = state.get("normalized_text") or ocr_result.get("full_text") or state.get("document_text") or ""
    ocr_confidence = _extract_ocr_confidence(ocr_result)
    # No `layout` key exists in real ocr_agent output today -- kept as a
    # separate extraction point (rather than deleted) so a future ocr_agent
    # version that does add structured layout info only needs a one-line
    # change here, not a change to the Qwen prompt plumbing.
    layout = [p.get("layout") for p in ocr_pages if isinstance(p, dict) and p.get("layout")] or None
    document_id = state.get("document_id") or ocr_result.get("document_id")

    image_bytes = _extract_image_bytes(state)

    if not normalized_text.strip() and image_bytes is None:
        raise MissingInputError("classification_agent: no OCR text and no document image available in state")

    return normalized_text, ocr_confidence, layout, ocr_pages, image_bytes, document_id


def _extract_ocr_confidence(ocr_result: Dict[str, Any]) -> float | None:
    """Real ocr_agent output has no per-item confidence score today (see
    docstring on _extract_inputs) -- this returns None in that case rather
    than guessing. Kept as its own function (not deleted) so a future
    ocr_agent version that does add confidence only needs this function
    updated, not every caller."""
    text_items = []
    for page in ocr_result.get("pages") or []:
        if not isinstance(page, dict):
            continue
        text_items.extend(page.get("text_items") or [])
    confidences = [ti.get("confidence") for ti in text_items if isinstance(ti, dict) and ti.get("confidence") is not None]
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def _extract_image_bytes(state: Dict[str, Any]) -> bytes | None:
    """Best-effort: state may carry a pre-rendered page image (produced by
    ocr_agent's pdf_renderer for its own OCR pass) as raw bytes or base64.
    classification_agent does not render PDFs itself -- rendering belongs
    to Agents/ocr_agent/processing/pdf_renderer.py, reused here rather than
    duplicated, per the "no invented new machinery" constraint.
    """
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


def _merge_result(state: Dict[str, Any], result: ClassificationResult) -> Dict[str, Any]:
    state["classification_result"] = result.to_dict()
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