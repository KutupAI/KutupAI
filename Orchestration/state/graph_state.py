"""
graph_state.py
----------------
Centralized, shared workflow state passed between the Supervisor and every
Agent adapter.

This is the ONE state contract for the whole Orchestration layer. Agents
read whatever section(s) they need and are expected to only ever write back
their own result section (enforced by StateManager, not by this module).

Backwards compatibility:
    The original OCR-only fields (document_id, document_path, question,
    ocr_result, ocr_status, ...) are preserved unchanged so the existing
    `process_service.run_workflow` keeps working without modification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class StageRecord(TypedDict, total=False):
    """One entry in the workflow execution history."""

    stage: str
    status: str  # completed | failed | invalid_result | missing_state | exception | skipped
    attempt: int
    started_at: str
    finished_at: str
    error: Optional[str]


class ErrorRecord(TypedDict, total=False):
    """A structured, non-sensitive error entry."""

    stage: str
    error_type: str
    message: str
    timestamp: str


class GraphState(TypedDict, total=False):
    """Mutable workflow state. Agents merge their outputs into this dict."""

    # --- Correlation / identity -------------------------------------------------
    workflow_id: str
    document_id: str
    request: Dict[str, Any]

    # --- Input -------------------------------------------------------------------
    document_path: str
    document_text: str
    accompanying_text: str
    question: str
    text: str

    # --- Per-stage results (Agents only ever write their own section) ------------
    ocr: Dict[str, Any]
    ocr_result: Dict[str, Any]  # unified { Success, Data } contract (legacy name)
    ocr_status: str

    classification: Dict[str, Any]
    classification_result: Dict[str, Any]
    classification_status: str

    extraction: Dict[str, Any]
    extraction_result: Dict[str, Any]
    extraction_status: str

    validation: Dict[str, Any]
    validation_result: Dict[str, Any]
    validation_status: str

    rag: Dict[str, Any]
    rag_query: str
    rag_result: Dict[str, Any]
    rag_status: str

    summary: Dict[str, Any]
    summary_text: str
    summary_status: str

    routing: Dict[str, Any]
    routing_decision: Dict[str, Any]
    routing_status: str

    writing: Dict[str, Any]
    draft_letter: str  # legacy compatibility; new writers use ``writing``.
    writing_status: str

    # --- Workflow control ----------------------------------------------------------
    current_agent: Optional[str]
    stage_status: Dict[str, str]
    stage_retries: Dict[str, int]
    history: List[StageRecord]
    errors: List[ErrorRecord]
    terminated: bool
    termination_reason: Optional[str]
    final_decision: Dict[str, Any]
    created_at: str
    updated_at: str


STATE_SECTION_BY_STAGE: Dict[str, str] = {
    "ocr": "ocr_result",
    "classification": "classification_result",
    "extraction": "extraction_result",
    "validation": "validation_result",
    "rag": "rag_result",
    "summary": "summary",
    "routing": "routing",
    "writing": "writing",
}
"""Maps a workflow stage name to the GraphState key that stage is allowed
to write. Used by StateManager to enforce "agents only write their own
section"."""


def empty_state() -> GraphState:
    """Return a freshly-initialized, empty GraphState skeleton."""

    return GraphState(
        stage_status={},
        stage_retries={},
        history=[],
        errors=[],
        terminated=False,
        termination_reason=None,
    )
