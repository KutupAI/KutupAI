"""OCR Agent — document → structured OCR for Orchestration.

Pipeline per page:
  quality → preprocess → PaddleOCR (Good/Usable?)
  → RapidOCR only if empty → PaddleOCR-VL if still bad
  → PP-StructureV3 tables only if table detected.

Writes state["ocr"] (+ wire keys ocr_result / ocr_status / document_text).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from Agents.base.agent_registry import register
from Agents.base.base_agent import BaseAgent
from Agents.ocr_agent.client import OCRClient, OCRRequest
from Agents.ocr_agent.config import OCRConfig
from Agents.ocr_agent.models import contract_envelope, empty_document, normalize_document

logger = logging.getLogger(__name__)

_DEFAULT_STORAGE_UPLOADS_DIR = "Storage/files/uploads"


@register
class OCRAgent(BaseAgent):
    """OCR only — no classification, RAG, or Storage writes."""

    name = "ocr_agent"
    description = (
        "Convert PDF/image/DOCX/PPTX/XLSX to structured OCR text "
        "(PaddleOCR primary; RapidOCR / PaddleOCR-VL / StructureV3 on demand)."
    )

    def __init__(self, client: OCRClient | None = None, config: OCRConfig | None = None) -> None:
        self.client = client or OCRClient(config)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(state)

        document = _extract_document(state)
        document_id = (document or {}).get("document_id") or state.get("document_id")
        file_path = _resolve_path(state, document)
        question = _extract_question(state)

        if not file_path:
            logger.error(
                "ocr_agent: could not resolve a file path for document_id=%s "
                "(state.request.document=%s)", document_id, document,
            )
            internal = _missing_path_internal(
                document_id=document_id,
                file_name=(document or {}).get("file_name"),
                file_type=(document or {}).get("file_type"),
                message="missing/unresolvable document in state.request.document",
            )
            return _apply(updated, internal, document=document, question=question)

        try:
            internal = self.client.process(OCRRequest(str(file_path), document_id=document_id))
        except Exception as exc:
            # An Agent must never crash the Supervisor's graph: report the
            # failure through state and let routing_logic decide what's next.
            logger.exception("ocr_agent: unhandled exception for document_id=%s", document_id)
            path = Path(str(file_path))
            internal = {
                "success": False,
                "status": "failed",
                "error": {"code": "OCR_FAILED", "message": str(exc)},
                "data": {
                    "document_id": document_id,
                    "file_name": (document or {}).get("file_name") or path.name,
                    "file_type": (document or {}).get("file_type") or path.suffix.lstrip("."),
                    "page_count": 0,
                    "language": {"detected": None},
                    "pages": [],
                    "full_text": "",
                },
            }

        return _apply(updated, internal, document=document, question=question)


# ----------------------------------------------------------------------
# Unified contract <-> internal OCRProcessor contract adapters
# ----------------------------------------------------------------------
def _apply(
    updated: Dict[str, Any],
    internal: Dict[str, Any],
    *,
    document: Dict[str, Any] | None,
    question: str,
) -> Dict[str, Any]:
    """Write both the unified agent section and the Orchestration wire keys."""

    updated["ocr"] = _to_unified_contract(internal)
    updated["ocr_result"] = _to_wire_envelope(internal, document=document, question=question)
    updated["ocr_status"] = _orchestration_status(internal)

    data = internal.get("data") or {}
    full_text = str(data.get("full_text") or "")
    if internal.get("success") and full_text.strip():
        updated["document_text"] = full_text

    if not internal.get("success"):
        errors = list(updated.get("errors") or [])
        err = internal.get("error") or {}
        msg = err.get("message") or "ocr failed"
        errors.append(f"ocr_agent: {msg}")
        updated["errors"] = errors

    return updated


def _extract_document(state: Dict[str, Any]) -> Dict[str, Any] | None:
    request = state.get("request")
    if isinstance(request, dict):
        document = request.get("document")
        if isinstance(document, dict):
            return document
    # Backward-compat: some older callers put `document` at the top level.
    document = state.get("document")
    if isinstance(document, dict):
        return document
    return None


def _extract_question(state: Dict[str, Any]) -> str:
    request = state.get("request")
    if isinstance(request, dict) and request.get("question"):
        return str(request.get("question") or "")
    for key in ("question", "accompanying_text", "text"):
        value = state.get(key)
        if value:
            return str(value)
    return ""


def _to_unified_contract(internal: Dict[str, Any]) -> Dict[str, Any]:
    """Map the internal OCRProcessor contract to the unified `state["ocr"]` shape."""
    data = internal.get("data") or {}
    pages = data.get("pages") or []

    language = data.get("language")
    if isinstance(language, dict):
        language = language.get("detected")

    contract: Dict[str, Any] = {
        "success": bool(internal.get("success")),
        "ocr_data": {
            "page_count": data.get("page_count", 0),
            "language": language,
            "pages": pages,
            "full_text": data.get("full_text", ""),
            "vision": _aggregate_vision(pages),
        },
    }
    status = internal.get("status")
    if status:
        contract["status"] = status
    if not internal.get("success"):
        err = internal.get("error") or {}
        contract["error"] = {
            "code": err.get("code", "OCR_FAILED"),
            "message": err.get("message", "ocr failed"),
        }
    return contract


def _to_wire_envelope(
    internal: Dict[str, Any],
    *,
    document: Dict[str, Any] | None,
    question: str,
) -> Dict[str, Any]:
    """Application/Orchestration wire contract: { Success, Data: [document] }."""
    data = internal.get("data") or {}
    doc = empty_document(
        document_id=str(data.get("document_id") or (document or {}).get("document_id") or ""),
        file_name=str(data.get("file_name") or (document or {}).get("file_name") or ""),
        file_type=str(data.get("file_type") or (document or {}).get("file_type") or ""),
        question=question,
    )
    doc["full_text"] = str(data.get("full_text") or "")
    doc["pages"] = list(data.get("pages") or [])
    doc = normalize_document(doc)
    return contract_envelope(bool(internal.get("success")), [doc])


def _orchestration_status(internal: Dict[str, Any]) -> str:
    """Map processor status to Orchestration's ocr_status vocabulary."""
    if not internal.get("success"):
        return "failed"
    status = str(internal.get("status") or "").lower()
    if status == "partial":
        return "partial"
    return "completed"


def _aggregate_vision(pages: list[dict[str, Any]]) -> Dict[str, Any]:
    """Document-level signature/stamp summary, OR-ed across pages."""
    signature_detected = False
    signature_handwritten = False
    stamp_detected = False
    for page in pages:
        vision = (page or {}).get("vision") or {}
        sig = vision.get("signature") or {}
        stamp = vision.get("stamp") or {}
        signature_detected = signature_detected or bool(sig.get("detected"))
        signature_handwritten = signature_handwritten or bool(sig.get("handwritten"))
        stamp_detected = stamp_detected or bool(stamp.get("detected"))
    return {
        "signature": {"detected": signature_detected, "handwritten": signature_handwritten},
        "stamp": {"detected": stamp_detected},
    }


def _missing_path_internal(
    *,
    document_id: Any,
    file_name: Any,
    file_type: Any,
    message: str,
) -> Dict[str, Any]:
    return {
        "success": False,
        "status": "failed",
        "error": {"code": "FILE_CORRUPTED", "message": message},
        "data": {
            "document_id": document_id,
            "file_name": file_name,
            "file_type": file_type,
            "page_count": 0,
            "language": {"detected": None},
            "pages": [],
            "full_text": "",
        },
    }


# ----------------------------------------------------------------------
# File resolution: Storage lookup by document_id/file_name, with
# backward-compatible support for the older direct-path state shapes.
# ----------------------------------------------------------------------
def _resolve_path(state: Dict[str, Any], document: Dict[str, Any] | None) -> str | None:
    # 1) Backward-compatible direct-path keys (older Orchestration state).
    for key in ("file_path", "document_path", "input_path"):
        value = state.get(key)
        if value:
            return str(value)
    if isinstance(document, dict):
        for key in ("file_path", "path", "document_path"):
            value = document.get(key)
            if value:
                return str(value)

    # 2) Unified contract: resolve document_id/file_name via Storage.
    if isinstance(document, dict):
        resolved = _resolve_via_storage(document)
        if resolved:
            return resolved
    return None


def _resolve_via_storage(document: Dict[str, Any]) -> str | None:
    document_id = document.get("document_id")
    file_name = document.get("file_name")

    # Prefer the project's real Storage resolver when it's importable, so
    # this Agent stays correct if Storage's layout ever changes.
    try:
        from Storage.file_locator import resolve_upload_path  # type: ignore

        resolved = resolve_upload_path(document_id=document_id, file_name=file_name)
        if resolved:
            return str(resolved)
    except Exception:
        pass

    if not file_name:
        return None
    uploads_dir = os.getenv("OCR_STORAGE_UPLOADS_DIR", _DEFAULT_STORAGE_UPLOADS_DIR)
    candidate = Path(uploads_dir) / str(file_name)
    return str(candidate)
