"""
Application-facing Orchestration entry point.

Two functions are exposed:

  * `run_ocr_pipeline`   - the original, OCR-only in-process step
                            (Application → OCRAgent). Unchanged behavior;
                            kept for backward compatibility with existing
                            Application/Presentation callers.
  * `run_full_workflow`  - runs the complete 8-stage graph (OCR ->
                            Classification -> Extraction -> Validation ->
                            RAG -> Summary -> Routing -> Writing) through
                            the Orchestration workflow engine. Stages whose
                            Agent isn't enabled/integrated yet in
                            config.yaml are skipped (never faked), so this
                            is safe to call today - it currently behaves
                            like `run_ocr_pipeline` plus a structured
                            final_decision, and will automatically exercise
                            more stages as Agents are connected one by one.

Both are kept free of FastAPI so unit/integration tests can import them
without installing the HTTP stack.

`run_ocr_pipeline` returns the unified layer contract:
  { "Success": bool, "Data": [ document, ... ] }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover
    from Agents.ocr_agent import OCRAgent

from Orchestration.graph.graph_definition import Stage
from Orchestration.messages.message_schema import (
    contract_envelope,
    empty_document,
    is_contract_envelope,
    normalize_document,
)
from Orchestration.workflow.workflow_builder import AgentProtocol, build_workflow
from Orchestration.workflow.workflow_config import WorkflowConfig

logger = logging.getLogger("Orchestration.process_service")


def run_ocr_pipeline(
    *,
    document_id: str,
    document_path: str | None,
    accompanying_text: str | None = None,
    agent: "OCRAgent | None" = None,
) -> dict[str, Any]:
    """Run OCRAgent against a temp file path from Application.

    The Agents.ocr_agent import is deferred to call time (rather than
    module import time) so this module - and anything importing it, e.g.
    `run_full_workflow` below - stays importable and unit-testable even in
    environments where the Agents/ layer isn't installed (as long as an
    `agent` instance or an `agent_overrides` mapping is supplied).
    """
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

    if agent is None:
        from Agents.ocr_agent import OCRAgent  # deferred, see docstring above

        worker = OCRAgent()
    else:
        worker = agent
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


def run_full_workflow(
    *,
    document_id: str,
    document_path: Optional[str],
    accompanying_text: Optional[str] = None,
    agent_overrides: Optional[Dict[Stage, AgentProtocol]] = None,
    config: Optional[WorkflowConfig] = None,
) -> Dict[str, Any]:
    """Run the full Orchestration graph (OCR through Writing) for one
    request and return the unified { Success, Data } contract.

    `Data[0]` is the OCR document contract (unchanged shape, so existing
    consumers of `run_ocr_pipeline` can switch over without a schema
    change) extended with the later-stage results when their Agents are
    integrated and enabled.
    """

    question = accompanying_text or ""
    request = {
        "document_id": document_id,
        "document_path": document_path,
        "accompanying_text": question or None,
        "question": question or None,
        "text": question or None,
    }

    workflow = build_workflow(config=config, agent_overrides=agent_overrides)
    result = workflow.run(request)
    state = result.state

    ocr_payload = state.get("ocr_result")
    if is_contract_envelope(ocr_payload):
        data = ocr_payload.get("Data") or []
        docs = [normalize_document(item) for item in data if isinstance(item, dict)]
        doc = docs[0] if docs else empty_document(document_id=document_id, question=question)
    else:
        doc = empty_document(document_id=document_id, question=question)

    for extra_key, state_key in (
        ("classification", "classification_result"),
        ("extraction", "extraction_result"),
        ("validation", "validation_result"),
        ("rag", "rag_result"),
        ("summary", "summary"),
        ("routing", "routing_decision"),
        ("writing", "draft_letter"),
    ):
        value = state.get(state_key)
        if value:
            doc[extra_key] = value

    success = result.completed and not result.terminated
    logger.info(
        "full_workflow_done workflow_id=%s document_id=%s success=%s stages_run=%s",
        state.get("workflow_id"),
        document_id,
        success,
        len(state.get("history", [])),
    )
    return contract_envelope(success, [doc])
