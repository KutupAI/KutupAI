"""
message_schema.py
------------------
Internal Orchestration message/result contracts, plus the unified
processing contract forwarded to Application/Presentation.

Two separate concerns live here:

1. External wire contract (unchanged, pass-through):
     { "Success": bool, "Data": [ document, ... ] }
   Produced by Agents/ocr_agent; Orchestration re-exports the same helpers
   so callers only ever import from Orchestration.messages.message_schema.

2. Internal orchestration contracts (new):
   AgentRequest / AgentResult / ExecutionStatus describe a single Agent
   invocation inside the workflow graph. These never leave Orchestration.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 1. External wire contract (pass-through from Agents/ocr_agent.models)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly whenever Agents/ is present
    from Agents.ocr_agent.models import (
        DOCUMENT_CONTRACT_KEYS,
        contract_envelope,
        empty_document,
        is_contract_envelope,
        normalize_document,
    )
except ImportError:  # pragma: no cover - fallback for isolated Orchestration testing
    # Agents/ is a sibling layer and may not be installed in every environment
    # that only needs to test the Orchestration graph (e.g. CI running unit
    # tests with mock Agents). This fallback mirrors the exact same
    # { Success, Data } wire contract so Orchestration remains importable
    # and testable on its own. When Agents.ocr_agent is available it is
    # always preferred (see try-block above) and this branch is unused.
    DOCUMENT_CONTRACT_KEYS = (
        "document_id",
        "file_name",
        "file_type",
        "full_text",
        "pages",
        "question",
        "answer",
    )

    def empty_document(
        *,
        document_id: str = "",
        file_name: str = "",
        file_type: str = "",
        question: str = "",
    ) -> Dict[str, Any]:
        return {
            "document_id": document_id,
            "file_name": file_name,
            "file_type": file_type,
            "full_text": "",
            "pages": [],
            "question": question,
            "answer": "",
        }

    def normalize_document(item: Dict[str, Any]) -> Dict[str, Any]:
        doc = empty_document()
        doc.update({k: v for k, v in item.items() if k in DOCUMENT_CONTRACT_KEYS})
        return doc

    def contract_envelope(success: bool, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"Success": bool(success), "Data": list(data)}

    def is_contract_envelope(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and "Success" in payload
            and isinstance(payload.get("Data"), list)
        )


# ---------------------------------------------------------------------------
# 2. Internal orchestration contracts
# ---------------------------------------------------------------------------
class ExecutionStatus(str, enum.Enum):
    """Outcome of a single Agent execution attempt."""

    SUCCESS = "success"
    FAILURE = "failure"
    INVALID_RESULT = "invalid_result"
    MISSING_STATE = "missing_state"
    EXCEPTION = "exception"
    SKIPPED = "skipped"
    NOT_INTEGRATED = "not_integrated"


@dataclass(frozen=True)
class AgentRequest:
    """What the Orchestration hands to an Agent adapter for one stage."""

    stage: str
    workflow_id: str
    attempt: int
    state: Dict[str, Any]


@dataclass
class AgentResult:
    """What an Agent adapter hands back to the Orchestration."""

    stage: str
    status: ExecutionStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS

    @classmethod
    def ok(cls, stage: str, data: Optional[Any] = None, **meta: Any) -> "AgentResult":
        return cls(stage=stage, status=ExecutionStatus.SUCCESS, data=data or {}, metadata=meta)

    @classmethod
    def fail(
        cls,
        stage: str,
        status: ExecutionStatus,
        error: str,
        error_type: Optional[str] = None,
        **meta: Any,
    ) -> "AgentResult":
        return cls(
            stage=stage,
            status=status,
            error=error,
            error_type=error_type or status.value,
            metadata=meta,
        )


__all__ = [
    "DOCUMENT_CONTRACT_KEYS",
    "contract_envelope",
    "empty_document",
    "is_contract_envelope",
    "normalize_document",
    "ExecutionStatus",
    "AgentRequest",
    "AgentResult",
]
