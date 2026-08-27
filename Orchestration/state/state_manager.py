"""
state_manager.py
-----------------
The ONLY write point for workflow state and Agent results.

Responsibilities:
  - initialize a fresh GraphState for an incoming request
  - merge an Agent's AgentResult into its own state section (and nowhere
    else - this is how "Agents only produce their own result section" is
    enforced even though Agents are plain Python callables)
  - track per-stage status, retry counters and execution history
  - record structured, non-sensitive errors
  - validate state shape before/after each transition
  - produce a redacted snapshot suitable for logging

No Agent, Supervisor, or workflow code should mutate GraphState directly;
everything goes through StateManager so there is exactly one place that
understands the state contract.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from Orchestration.graph.graph_definition import Stage
from Orchestration.messages.message_schema import AgentResult, ExecutionStatus, is_contract_envelope
from Orchestration.state.graph_state import (
    STATE_SECTION_BY_STAGE,
    GraphState,
    empty_state,
)

logger = logging.getLogger("Orchestration.state_manager")


def _aggregate_vision_from_pages(pages: Any) -> Dict[str, Any]:
    signature_detected = False
    signature_handwritten = False
    stamp_detected = False
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict):
                continue
            vision = page.get("vision") if isinstance(page.get("vision"), dict) else {}
            sig = vision.get("signature") if isinstance(vision.get("signature"), dict) else {}
            stamp = vision.get("stamp") if isinstance(vision.get("stamp"), dict) else {}
            signature_detected = signature_detected or bool(sig.get("detected"))
            signature_handwritten = signature_handwritten or bool(sig.get("handwritten"))
            stamp_detected = stamp_detected or bool(stamp.get("detected"))
    return {
        "signature": {"detected": signature_detected, "handwritten": signature_handwritten},
        "stamp": {"detected": stamp_detected},
    }


def unified_ocr_from_wire(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Build Layers_contracts `ocr` slot from wire `{ Success, Data: [doc] }`."""
    data = envelope.get("Data") or envelope.get("data") or []
    doc = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else {}
    pages = doc.get("pages") if isinstance(doc.get("pages"), list) else []
    language = doc.get("language")
    if isinstance(language, dict):
        language = language.get("detected")
    return {
        "success": bool(envelope.get("Success", envelope.get("success", True))),
        "ocr_data": {
            "page_count": doc.get("page_count") or (len(pages) if pages else 0),
            "language": language or "tr",
            "pages": pages,
            "full_text": str(doc.get("full_text") or ""),
            "vision": _aggregate_vision_from_pages(pages),
        },
    }


def wire_ocr_from_unified(ocr: Dict[str, Any], *, document_id: str = "", question: str = "") -> Dict[str, Any]:
    """Build wire `{ Success, Data }` from unified `ocr` for process_service / legacy readers."""
    ocr_data = ocr.get("ocr_data") if isinstance(ocr.get("ocr_data"), dict) else {}
    doc = {
        "document_id": document_id,
        "full_text": str(ocr_data.get("full_text") or ""),
        "pages": list(ocr_data.get("pages") or []),
        "page_count": ocr_data.get("page_count") or 0,
        "language": ocr_data.get("language") or "tr",
        "question": question,
    }
    return {"Success": bool(ocr.get("success", True)), "Data": [doc]}

# Fields every valid GraphState must contain once initialized.
_REQUIRED_KEYS = (
    "workflow_id",
    "document_id",
    "stage_status",
    "stage_retries",
    "history",
    "errors",
)

# Application / Agents pipeline envelope stage sections (always present).
_STAGE_SECTION_KEYS = (
    "ocr",
    "classification",
    "extraction",
    "validation",
    "rag",
    "summary",
    "routing",
    "writing",
)

# State content that must never be written to logs.
_SENSITIVE_KEYS = frozenset(
    {
        "document_text",
        "full_text",
        "accompanying_text",
        "question",
        "text",
        "draft_letter",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateManager:
    """Owns the lifecycle of a single workflow's GraphState."""

    def __init__(self, workflow_id: Optional[str] = None) -> None:
        self.workflow_id = workflow_id or uuid.uuid4().hex

    # -- lifecycle ---------------------------------------------------------
    def initialize(self, request: Dict[str, Any]) -> GraphState:
        """Build the initial GraphState for one incoming request.

        Accepts either:
          * Application envelope fields mixed into the seed
            (``request`` = {success, question, document} plus optional
            empty ``ocr``/``classification``/… sections), or
          * a legacy flat dict (document_id / document_path / question).

        ``state["request"]`` is always normalized to the Application
        contract shape. Stage sections start as ``{}`` unless the caller
        already supplied them. Flat fields are still lifted to the top
        level so OCR path resolution keeps working.
        """

        nested = request.get("request")
        if isinstance(nested, dict) and (
            "document" in nested or "question" in nested or "success" in nested
        ):
            request_section = dict(nested)
            document = (
                dict(request_section["document"])
                if isinstance(request_section.get("document"), dict)
                else {}
            )
            document_id = str(
                document.get("document_id")
                or request.get("document_id")
                or uuid.uuid4().hex
            )
            document.setdefault("document_id", document_id)
            request_section["document"] = document
            request_section.setdefault("success", True)
            request_section.setdefault(
                "question",
                str(
                    request.get("question")
                    or request.get("accompanying_text")
                    or request.get("text")
                    or ""
                ),
            )
        else:
            document_id = str(request.get("document_id") or uuid.uuid4().hex)
            document = (
                dict(request["document"])
                if isinstance(request.get("document"), dict)
                else {}
            )
            document.setdefault("document_id", document_id)
            if request.get("document_path") and not document.get("document_path"):
                document["document_path"] = request["document_path"]
            request_section = {
                "success": bool(request.get("success", True)),
                "question": str(
                    request.get("question")
                    or request.get("accompanying_text")
                    or request.get("text")
                    or ""
                ),
                "document": document,
            }

        state: GraphState = empty_state()
        state.update(
            {
                "workflow_id": self.workflow_id,
                "document_id": document_id,
                "request": request_section,
                "created_at": _now(),
                "updated_at": _now(),
                "current_agent": None,
                "final_decision": {},
            }
        )

        for key in _STAGE_SECTION_KEYS:
            value = request.get(key)
            state[key] = dict(value) if isinstance(value, dict) else {}  # type: ignore[literal-required]

        for key in (
            "document_path", "file_name", "file_type", "accompanying_text", "question", "text",
            "conversation_memory", "conversation_focus_law", "conversation_reference_law",
            "conversation_is_follow_up", "ocr_cache_hit", "document_upload_only", "writer_instruction",
        ):
            value = request.get(key)
            if not value and key == "question":
                value = request_section.get("question")
            if not value and key in ("document_path", "file_name", "file_type"):
                value = (request_section.get("document") or {}).get(key)
            if value:
                state[key] = value

        missing = self.validate(state)
        if missing:
            # Should never happen right after initialize(); guard anyway.
            logger.warning(
                "state_initialize_incomplete workflow_id=%s missing=%s",
                self.workflow_id,
                missing,
            )
        logger.info(
            "state_initialized workflow_id=%s document_id=%s",
            self.workflow_id,
            document_id,
        )
        return state

    def validate(self, state: GraphState) -> List[str]:
        """Return the list of required keys missing from `state`."""

        return [key for key in _REQUIRED_KEYS if key not in state]

    # -- writes --------------------------------------------------------------
    def begin_attempt(self, state: GraphState, stage: Stage, attempt: int) -> None:
        state["current_agent"] = stage.value
        state["stage_status"][stage.value] = "running"
        state["updated_at"] = _now()
        logger.info(
            "agent_start workflow_id=%s stage=%s attempt=%s",
            self.workflow_id,
            stage.value,
            attempt,
        )

    def apply_result(self, state: GraphState, stage: Stage, result: AgentResult, attempt: int) -> GraphState:
        """The single write point: merge one Agent's result into its own
        section of GraphState, update status/history, and never touch any
        other stage's section."""

        started = state.get("updated_at", _now())
        state["updated_at"] = _now()
        state["stage_status"][stage.value] = result.status.value

        if result.status == ExecutionStatus.SUCCESS and result.data is not None:
            section_key = STATE_SECTION_BY_STAGE.get(stage.value)
            if section_key:
                state[section_key] = result.data  # type: ignore[literal-required]
            # Dual-key mirror: keep the unified short section alongside the
            # Orchestration wire key (classification / validation / …).
            if stage == Stage.OCR:
                if isinstance(result.data, dict) and is_contract_envelope(result.data):
                    # Mock / legacy Agents still return the wire envelope as data.
                    state["ocr_result"] = result.data  # type: ignore[literal-required]
                    state["ocr"] = unified_ocr_from_wire(result.data)  # type: ignore[literal-required]
                    data_list = result.data.get("Data") or []
                    if isinstance(data_list, list) and data_list and isinstance(data_list[0], dict):
                        full_text = str(data_list[0].get("full_text") or "")
                        if full_text.strip():
                            state["document_text"] = full_text
                elif isinstance(result.data, dict):
                    state["ocr"] = result.data  # type: ignore[literal-required]
                    # Preserve wire mirror so process_service can still seed Data[0].
                    if not is_contract_envelope(state.get("ocr_result")):
                        state["ocr_result"] = wire_ocr_from_unified(  # type: ignore[literal-required]
                            result.data,
                            document_id=str(state.get("document_id") or ""),
                            question=str(state.get("question") or ""),
                        )
                    ocr_data = result.data.get("ocr_data") if isinstance(result.data.get("ocr_data"), dict) else {}
                    full_text = str(ocr_data.get("full_text") or "")
                    if full_text.strip():
                        state["document_text"] = full_text
                state["ocr_status"] = "completed"
            elif stage == Stage.CLASSIFICATION:
                state["classification"] = result.data  # type: ignore[literal-required]
                state["classification_status"] = "completed"
            elif stage == Stage.EXTRACTION:
                state["extraction"] = result.data  # type: ignore[literal-required]
                state["extraction_status"] = "completed"
            elif stage == Stage.VALIDATION:
                state["validation"] = result.data  # type: ignore[literal-required]
            elif stage == Stage.RAG:
                # Prefer unified short slot; keep rag_result in SummaryAgent RAGResult shape.
                state["rag"] = result.data  # type: ignore[literal-required]
                if isinstance(result.data, dict) and "rag_data" in result.data:
                    state["rag_result"] = {
                        "success": bool(result.data.get("success")),
                        "data": result.data.get("rag_data"),
                        "error": result.data.get("error"),
                    }
                state["rag_status"] = (
                    "completed" if isinstance(result.data, dict) and result.data.get("success") else "failed"
                )
        elif stage == Stage.RAG and isinstance(result.data, dict) and result.data:
            # Persist soft/hard RAG failures so Presentation never sees rag:{}.
            state["rag"] = result.data  # type: ignore[literal-required]
            if "rag_data" in result.data:
                state["rag_result"] = {
                    "success": bool(result.data.get("success")),
                    "data": result.data.get("rag_data"),
                    "error": result.data.get("error"),
                }
            state["rag_status"] = "failed"

        if not result.is_success:
            self.record_error(
                state,
                stage=stage,
                error_type=result.error_type or result.status.value,
                message=result.error or f"{stage.value} did not complete successfully",
            )

        state["history"].append(
            {
                "stage": stage.value,
                "status": result.status.value,
                "attempt": attempt,
                "started_at": started,
                "finished_at": state["updated_at"],
                "error": result.error,
            }
        )

        logger.info(
            "agent_done workflow_id=%s stage=%s status=%s attempt=%s",
            self.workflow_id,
            stage.value,
            result.status.value,
            attempt,
        )
        return state

    def increment_retry(self, state: GraphState, stage: Stage) -> int:
        count = state["stage_retries"].get(stage.value, 0) + 1
        state["stage_retries"][stage.value] = count
        return count

    def record_error(self, state: GraphState, *, stage: Stage, error_type: str, message: str) -> None:
        state["errors"].append(
            {
                "stage": stage.value,
                "error_type": error_type,
                "message": message,
                "timestamp": _now(),
            }
        )
        logger.error(
            "agent_error workflow_id=%s stage=%s error_type=%s message=%s",
            self.workflow_id,
            stage.value,
            error_type,
            message,
        )

    def terminate(self, state: GraphState, reason: str) -> GraphState:
        state["terminated"] = True
        state["termination_reason"] = reason
        state["updated_at"] = _now()
        logger.warning(
            "workflow_terminated workflow_id=%s reason=%s",
            self.workflow_id,
            reason,
        )
        return state

    def finalize(self, state: GraphState, final_decision: Dict[str, Any]) -> GraphState:
        state["final_decision"] = final_decision
        state["current_agent"] = None
        state["updated_at"] = _now()
        logger.info(
            "workflow_complete workflow_id=%s terminated=%s stages_run=%s",
            self.workflow_id,
            state.get("terminated", False),
            len(state.get("history", [])),
        )
        return state

    # -- read-only helpers -----------------------------------------------------
    @staticmethod
    def snapshot(state: GraphState) -> Dict[str, Any]:
        """Redacted, log-safe view of the state (no document content)."""

        return {
            key: ("<redacted>" if key in _SENSITIVE_KEYS and value else value)
            for key, value in state.items()
        }
