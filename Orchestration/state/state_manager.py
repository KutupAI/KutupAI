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
from Orchestration.messages.message_schema import AgentResult, ExecutionStatus
from Orchestration.state.graph_state import (
    STATE_SECTION_BY_STAGE,
    GraphState,
    empty_state,
)

logger = logging.getLogger("Orchestration.state_manager")

# Fields every valid GraphState must contain once initialized.
_REQUIRED_KEYS = (
    "workflow_id",
    "document_id",
    "stage_status",
    "stage_retries",
    "history",
    "errors",
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

        `request` is whatever the caller (process_service / main.py)
        received from Application; it is copied into `state["request"]`
        and its recognizable fields are lifted to the top level so
        existing Agents (e.g. OCRAgent, which expects document_id /
        document_path / question at the top level) keep working unchanged.
        """

        document_id = str(request.get("document_id") or uuid.uuid4().hex)
        state: GraphState = empty_state()
        state.update(
            {
                "workflow_id": self.workflow_id,
                "document_id": document_id,
                "request": dict(request),
                "created_at": _now(),
                "updated_at": _now(),
                "current_agent": None,
                "final_decision": {},
            }
        )
        for key in ("document_path", "accompanying_text", "question", "text"):
            value = request.get(key)
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
            # Legacy/back-compat mirror for the OCR stage only.
            if stage == Stage.OCR and "ocr_status" not in result.data:
                state["ocr_status"] = "completed"

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
