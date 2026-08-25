"""
rag_agent
---------
Retrieve-only legal evidence for the Orchestration workflow.

Calls ``RAG.client.handle_layer_state`` (never LLM generation). Writes the
Layers_contracts short slot ``rag`` and the Orchestration wire mirror
``rag_result`` / ``rag_status``. Does not call Storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from Agents.base.agent_registry import register
from Agents.base.base_agent import BaseAgent
from Agents.rag_agent.tools import run_layer_retrieve

# Canonical keys written to state["rag"] (unified Layers_contracts slot).
RAG_CONTRACT_KEYS = ("success", "rag_data")


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _ensure_envelope(state: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize GraphState / process_service request into Layers_contracts shape.

    ``handle_layer_state`` requires ``request.success`` and builds the query from
    ``request.question`` + classification + OCR text. Orchestration may only
    have top-level ``question`` / ``ocr_result`` / ``document_text``.
    """
    updated = dict(state)
    request = dict(_as_dict(updated.get("request")))
    if "success" not in request:
        request["success"] = True
    if not _clean(request.get("question")):
        request["question"] = (
            _clean(updated.get("question"))
            or _clean(updated.get("accompanying_text"))
            or _clean(updated.get("rag_query"))
            or _clean(updated.get("document_text"))
        )
    document = _as_dict(request.get("document"))
    if not document:
        path = Path(str(updated.get("document_path") or ""))
        document = {
            "document_id": str(updated.get("document_id") or ""),
            "file_name": path.name or str(updated.get("document_id") or ""),
            "file_type": path.suffix.lstrip(".") or "",
        }
        request["document"] = document
    updated["request"] = request

    ocr = _as_dict(updated.get("ocr"))
    ocr_data = _as_dict(ocr.get("ocr_data"))
    if not _clean(ocr_data.get("full_text")):
        full_text = _clean(updated.get("document_text")) or _clean(updated.get("text"))
        if not full_text:
            ocr_result = _as_dict(updated.get("ocr_result"))
            data = ocr_result.get("Data") or ocr_result.get("data")
            if isinstance(data, list) and data and isinstance(data[0], dict):
                full_text = _clean(data[0].get("full_text"))
            elif _clean(ocr_result.get("full_text")):
                full_text = _clean(ocr_result.get("full_text"))
        if full_text:
            updated["ocr"] = {
                "success": True,
                "ocr_data": {
                    "page_count": int(ocr_data.get("page_count") or 1),
                    "language": ocr_data.get("language") or "tr",
                    "pages": list(ocr_data.get("pages") or []),
                    "full_text": full_text,
                },
            }

    classification = _as_dict(updated.get("classification"))
    if not classification.get("success"):
        wire = _as_dict(updated.get("classification_result"))
        if wire.get("success") or wire.get("document_type"):
            updated["classification"] = {
                "success": bool(wire.get("success", True)),
                "document_type": wire.get("document_type") or "",
                "classification_confidence": wire.get("classification_confidence"),
            }

    return updated


def _wire_rag_result(rag_slot: Dict[str, Any]) -> Dict[str, Any]:
    """SummaryAgent / MockRagAgent RAGResult shape: {success, data, error?}."""
    return {
        "success": bool(rag_slot.get("success")),
        "data": rag_slot.get("rag_data"),
        "error": rag_slot.get("error"),
    }


@register
class RAGAgent(BaseAgent):
    name = "rag_agent"
    description = "Retrieve sourced legal passages for the document (no LLM answer)."

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(state, dict):
            raise TypeError("RAGAgent.run expects GraphState / envelope as a dict")

        envelope = _ensure_envelope(state)
        output = run_layer_retrieve(envelope)
        rag_slot = _as_dict(output.get("rag"))
        if not rag_slot:
            rag_slot = {
                "success": False,
                "rag_data": {"operation": "retrieve", "query": "", "results": []},
                "error": {"code": "empty_rag", "message": "RAG layer returned no rag section."},
            }

        updated = dict(output)
        updated["rag"] = rag_slot
        updated["rag_result"] = _wire_rag_result(rag_slot)
        # Always "completed" once a structured rag slot exists — empty/error
        # retrieval is a soft outcome for Orchestration (not a graph crash).
        updated["rag_status"] = "completed"
        return updated

    def process(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline envelope entry (same as run)."""
        return self.run(envelope)


# Backward-compatible alias used by older imports.
RagAgent = RAGAgent


def process(envelope: Dict[str, Any], agent: Optional[RAGAgent] = None) -> Dict[str, Any]:
    return (agent or RAGAgent()).process(envelope)
