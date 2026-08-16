"""
In-process Orchestration OCR step (Application → OCRAgent).

Kept free of FastAPI so unit/integration tests can import it without
installing the HTTP stack.

Returns the unified layer contract:
  { "Success": bool, "Data": [ document, ... ] }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from Agents.ocr_agent import OCRAgent
from Orchestration.messages.message_schema import (
    contract_envelope,
    empty_document,
    is_contract_envelope,
    normalize_document,
)

logger = logging.getLogger("Orchestration.process_service")


def run_ocr_pipeline(
    *,
    document_id: str,
    document_path: str | None,
    accompanying_text: str | None = None,
    agent: OCRAgent | None = None,
) -> dict[str, Any]:
    """Run OCRAgent against a temp file path from Application."""
    question = accompanying_text or ""

    if not document_path:
        logger.info("process_skip document_id=%s reason=missing_document_path", document_id)
        return contract_envelope(
            False,
            [empty_document(document_id=document_id, question=question)],
        )

    path = Path(document_path)
    if not path.is_file():
        logger.info("process_skip document_id=%s reason=path_missing", document_id)
        return contract_envelope(
            False,
            [
                empty_document(
                    document_id=document_id,
                    file_name=path.name,
                    file_type=path.suffix.lstrip("."),
                    question=question,
                )
            ],
        )

    state: dict[str, Any] = {
        "document_id": document_id,
        "document_path": str(path),
    }
    if question:
        state["accompanying_text"] = question
        state["text"] = question
        state["question"] = question

    worker = agent or OCRAgent()
    logger.info(
        "process_start document_id=%s path=%s has_text=%s",
        document_id,
        path.name,
        bool(question),
    )
    updated = worker.run(state)
    envelope = _as_envelope(
        updated.get("ocr_result"),
        document_id=document_id,
        file_name=path.name,
        file_type=path.suffix.lstrip("."),
        question=question,
        success=updated.get("ocr_status") == "completed",
    )
    if envelope["Data"]:
        doc = envelope["Data"][0]
        if not doc.get("document_id"):
            doc["document_id"] = document_id
        if question and not doc.get("question"):
            doc["question"] = question
        if not doc.get("file_name"):
            doc["file_name"] = path.name
        if not doc.get("file_type"):
            doc["file_type"] = path.suffix.lstrip(".")

    logger.info(
        "process_done document_id=%s success=%s pages=%s",
        document_id,
        envelope["Success"],
        len((envelope["Data"][0].get("pages") if envelope["Data"] else []) or []),
    )
    return envelope


def _as_envelope(
    payload: Any,
    *,
    document_id: str,
    file_name: str,
    file_type: str,
    question: str,
    success: bool,
) -> dict[str, Any]:
    if is_contract_envelope(payload):
        data = payload.get("Data") if isinstance(payload.get("Data"), list) else []
        docs = [normalize_document(item) for item in data if isinstance(item, dict)]
        if not docs:
            docs = [
                empty_document(
                    document_id=document_id,
                    file_name=file_name,
                    file_type=file_type,
                    question=question,
                )
            ]
        envelope_success = bool(payload.get("Success"))
        doc_preview = docs[0] if docs else {}
        if not envelope_success and (
            success
            or str(doc_preview.get("full_text") or "").strip()
            or str(doc_preview.get("answer") or "").strip()
        ):
            envelope_success = True
        return contract_envelope(envelope_success, docs)

    return contract_envelope(
        False,
        [
            empty_document(
                document_id=document_id,
                file_name=file_name,
                file_type=file_type,
                question=question,
            )
        ],
    )
