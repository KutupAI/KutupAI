"""
ocr_agent — converts document images/PDFs/office files to structured text
for Orchestration.

Pipeline: validate -> normalize -> page extraction -> quality analysis ->
adaptive preprocessing -> PaddleOCR/PP-StructureV3 -> confidence analysis
-> accept/retry/vision-fallback -> stable output contract.

Scope boundary (see README): OCR only. No classification, extraction,
validation, RAG, summarization, or writing happens here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from Agents.base.agent_registry import register
from Agents.base.base_agent import BaseAgent
from Agents.ocr_agent.client import OCRClient, OCRRequest
from Agents.ocr_agent.config import OCRConfig

logger = logging.getLogger(__name__)


@register
class OCRAgent(BaseAgent):
    """OCR only — no classification, RAG, Storage writes, or business logic."""

    name = "ocr_agent"
    description = (
        "Convert PDF/image/DOCX/PPTX/XLSX documents to structured OCR text "
        "(PaddleOCR + PP-StructureV3, Turkish/English), with adaptive "
        "preprocessing, confidence-based retries, and an optional vision "
        "fallback for otherwise-unreadable pages."
    )

    def __init__(self, client: OCRClient | None = None, config: OCRConfig | None = None) -> None:
        self.client = client or OCRClient(config)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(state)
        document_id = state.get("document_id")
        file_path = _resolve_path(state)

        if not file_path:
            contract = _missing_path_contract(document_id)
            return _apply(updated, contract)

        try:
            contract = self.client.process(OCRRequest(str(file_path), document_id=document_id))
        except Exception as exc:
            # An Agent must never crash the Supervisor's graph: report the
            # failure through state and let routing_logic decide what's next.
            logger.exception("ocr_agent: unhandled exception for document_id=%s", document_id)
            contract = {
                "success": False,
                "status": "failed",
                "error": {"code": "OCR_FAILED", "message": str(exc)},
                "data": {
                    "document_id": document_id,
                    "file_name": None,
                    "file_type": None,
                    "page_count": 0,
                    "language": {"detected": None},
                    "pages": [],
                    "full_text": "",
                },
            }

        return _apply(updated, contract)


def _apply(updated: Dict[str, Any], contract: Dict[str, Any]) -> Dict[str, Any]:
    updated["ocr_result"] = contract
    updated["ocr_status"] = contract.get("status", "failed")

    data = contract.get("data") or {}
    full_text = str(data.get("full_text") or "")
    if contract.get("success") and full_text.strip():
        # Backward/forward-compatible handoff for downstream Agents that
        # only need plain text (Classification, Extraction, RAG, Summary...).
        updated["document_text"] = full_text

    if not contract.get("success"):
        errors = list(updated.get("errors") or [])
        err = contract.get("error") or {}
        msg = err.get("message") or "ocr failed"
        errors.append(f"ocr_agent: {msg}")
        updated["errors"] = errors

    return updated


def _missing_path_contract(document_id: Any) -> Dict[str, Any]:
    return {
        "success": False,
        "status": "failed",
        "error": {"code": "FILE_CORRUPTED", "message": "missing document path in state"},
        "data": {
            "document_id": document_id,
            "file_name": None,
            "file_type": None,
            "page_count": 0,
            "language": {"detected": None},
            "pages": [],
            "full_text": "",
        },
    }


def _resolve_path(state: Dict[str, Any]) -> str | None:
    for key in ("file_path", "document_path", "input_path"):
        value = state.get(key)
        if value:
            return str(value)
    document = state.get("document")
    if isinstance(document, dict):
        for key in ("file_path", "path", "document_path"):
            value = document.get(key)
            if value:
                return str(value)
    return None
