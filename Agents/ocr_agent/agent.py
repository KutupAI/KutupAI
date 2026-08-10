"""
ocr_agent — converts document images/PDFs to text for Orchestration.

Pipeline: validate → load pages → preprocess → PaddleOCR/PP-StructureV3
→ parse/layout/tables → Turkish correction → UnifiedOCRResult
"""

from __future__ import annotations

from typing import Any, Dict

from Agents.base.agent_registry import register
from Agents.base.base_agent import BaseAgent
from Agents.ocr_agent.client import OCRClient, OCRRequest
from Agents.ocr_agent.config import OCRConfig
from Agents.ocr_agent.exceptions import OCRAgentError


@register
class OCRAgent(BaseAgent):
    """OCR only — no classification, RAG, Storage writes, or business logic."""

    name = "ocr_agent"
    description = "Convert PDF/image documents to text via PaddleOCR (Turkish)."

    def __init__(self, client: OCRClient | None = None, config: OCRConfig | None = None) -> None:
        self.client = client or OCRClient(config)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(state)
        file_path = _resolve_path(state)

        if not file_path:
            updated["ocr_result"] = {
                "success": False,
                "full_text": "",
                "error": "missing_document_path",
            }
            updated["ocr_status"] = "failed"
            errors = list(updated.get("errors") or [])
            errors.append("ocr_agent: missing document path")
            updated["errors"] = errors
            return updated

        try:
            result = self.client.process(
                OCRRequest(str(file_path), document_id=state.get("document_id"))
            )
        except OCRAgentError as exc:
            updated["ocr_result"] = {"success": False, "full_text": "", "error": str(exc)}
            updated["ocr_status"] = "failed"
            errors = list(updated.get("errors") or [])
            errors.append(f"ocr_agent: {exc}")
            updated["errors"] = errors
            return updated

        updated["ocr_result"] = result.to_dict()
        updated["ocr_status"] = "completed" if result.success else "failed"
        if result.success and result.full_text.strip():
            updated["document_text"] = result.full_text
        if not result.success:
            errors = list(updated.get("errors") or [])
            msg = result.errors[0].message if result.errors else "ocr failed"
            errors.append(f"ocr_agent: {msg}")
            updated["errors"] = errors
        return updated


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
