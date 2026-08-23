"""
Application-facing Orchestration entry point.

Single path: `run_workflow` runs the full stage graph
(OCR → Classification → … → Writing). Stages that are disabled in
config.yaml are skipped (never faked). Today only OCR is enabled, so the
graph still starts at OCRAgent like every other Agent stage.

Returns the unified layer contract: { "Success": bool, "Data": [ document, ... ] }.

Kept free of FastAPI so unit/integration tests can import without the HTTP stack.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

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


def run_workflow(
    *,
    document_id: str,
    document_path: Optional[str],
    accompanying_text: Optional[str] = None,
    agent_overrides: Optional[Dict[Stage, AgentProtocol]] = None,
    config: Optional[WorkflowConfig] = None,
) -> Dict[str, Any]:
    """Run the Orchestration graph for one request; return { Success, Data }."""

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

    request = {
        "document_id": document_id,
        "document_path": str(path),
        "accompanying_text": question or None,
        "question": question or None,
        "text": question or None,
    }

    logger.info(
        "workflow_start document_id=%s path=%s has_text=%s",
        document_id,
        path.name,
        bool(question),
    )

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

    if not doc.get("document_id"):
        doc["document_id"] = document_id
    if question and not doc.get("question"):
        doc["question"] = question
    if not doc.get("file_name"):
        doc["file_name"] = path.name
    if not doc.get("file_type"):
        doc["file_type"] = path.suffix.lstrip(".")

    for extra_key, state_key in (
        ("classification", "classification_result"),
        ("extraction", "extraction_result"),
        ("validation", "validation_result"),
        ("rag", "rag_result"),
        ("summary", "summary"),
        ("routing", "routing_decision"),
        ("writing", "writing"),
    ):
        value = state.get(state_key)
        if value:
            doc[extra_key] = value

    success = result.completed and not result.terminated
    # OCR-only configs: treat a successful OCR envelope as overall success
    # when later stages were skipped (not failed).
    if not success and is_contract_envelope(ocr_payload) and ocr_payload.get("Success"):
        success = True

    logger.info(
        "workflow_done workflow_id=%s document_id=%s success=%s stages_run=%s",
        state.get("workflow_id"),
        document_id,
        success,
        len(state.get("history", [])),
    )
    return contract_envelope(success, [doc])


# Backward-compatible name used by older imports/docs.
run_full_workflow = run_workflow
