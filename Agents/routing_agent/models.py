"""
models.py
=========

All data contracts used by the Routing Agent:

  * SharedStateInput  - the subset of the shared pipeline state the Routing
                         Agent consumes (does not redefine or break the
                         existing shared-state contract; only reads it).
  * Department         - a node in the routing knowledge base.
  * IntentSegment       - one detected intent/request inside a document.
  * ScoreBreakdown      - the per-signal scores that make up a candidate's
                         total score (kept internally, never exposed raw as
                         "chain of thought" -- only distilled into evidence).
  * RouteCandidate      - a scored department candidate, with positive and
                         negative evidence (negative routing).
  * Route               - a resolved, hierarchical route with an authority.
  * RoutingResult        - the final agent output contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

class RoutingStatus(str, Enum):
    SINGLE_ROUTE = "SINGLE_ROUTE"
    MULTI_ROUTE = "MULTI_ROUTE"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICT_DETECTED = "CONFLICT_DETECTED"
    ROUTING_CONFLICT = "ROUTING_CONFLICT"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ProcessingChannel(str, Enum):
    STANDARD = "STANDARD"
    URGENT = "URGENT"
    LEGAL = "LEGAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    EXTERNAL_CORRESPONDENCE = "EXTERNAL_CORRESPONDENCE"


# --------------------------------------------------------------------------
# Shared-state input contract
# --------------------------------------------------------------------------

@dataclass
class SharedStateInput:
    """The fields the Routing Agent reads from the existing shared state.

    Only ever *read*. The Routing Agent never mutates upstream fields and
    never assumes fields not listed in the specification are present.
    """

    document_text: str
    document_type: Optional[str] = None
    summary: Optional[str] = None
    intent: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    topic: Optional[str] = None
    subtopics: List[str] = field(default_factory=list)
    requested_action: Optional[str] = None
    sender: Optional[str] = None
    recipient: Optional[str] = None
    institution: Optional[str] = None
    legal_references: List[str] = field(default_factory=list)
    previous_correspondence: List[Dict[str, Any]] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    classification_confidence: Optional[float] = None
    analysis_confidence: Optional[float] = None
    writing_output: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SharedStateInput":
        """Tolerantly build an input from an arbitrary shared-state dict.

        Unknown keys are ignored; missing keys fall back to defaults. This
        keeps the agent decoupled from the exact shared-state object type
        used elsewhere in KutupAI.
        """
        known = {f: data.get(f) for f in cls.__dataclass_fields__ if f in data}
        # Normalize a couple of loose/aliased shapes defensively.
        if "entities" in known and known["entities"] is None:
            known["entities"] = []
        if "document_text" not in known or known.get("document_text") is None:
            known["document_text"] = data.get("text", "") or ""
        return cls(**known)

    @classmethod
    def from_envelope(cls, envelope: Dict[str, Any]) -> "SharedStateInput":
        """Build an input from the pipeline envelope contract:

            {request, ocr, classification, extraction, validation,
             rag, summary, routing, writing}

        Every lookup is defensive: any stage may be `{}`, missing, or have
        `success: false` -- the Routing Agent must still attempt a
        best-effort decision (never crash on partial upstream data).
        """
        def g(d: Optional[Dict[str, Any]], *path, default=None):
            cur = d or {}
            for key in path:
                if not isinstance(cur, dict):
                    return default
                cur = cur.get(key)
                if cur is None:
                    return default
            return cur if cur is not None else default

        request = envelope.get("request") or {}
        ocr = envelope.get("ocr") or {}
        classification = envelope.get("classification") or {}
        extraction = envelope.get("extraction") or {}
        validation = envelope.get("validation") or {}
        rag = envelope.get("rag") or {}
        summary = envelope.get("summary") or {}

        document_text = g(ocr, "ocr_data", "full_text", default="") or ""
        document_type = classification.get("document_type")
        question = request.get("question")
        summary_text = summary.get("rag_summary_text")

        rag_results = g(rag, "rag_data", "results", default=[]) or []
        rag_query = g(rag, "rag_data", "query")

        metadata: Dict[str, Any] = {
            "document_id": g(request, "document", "document_id"),
            "file_name": g(request, "document", "file_name"),
            "file_type": g(request, "document", "file_type"),
            "classification_topic": document_type,
            "analysis_summary": summary_text,
            "request_success": request.get("success"),
            "ocr_success": ocr.get("success"),
            "classification_success": classification.get("success"),
            "extraction_success": extraction.get("success"),
            "validation_success": validation.get("success"),
            "rag_success": rag.get("success"),
            "summary_success": summary.get("success"),
            "is_complete": validation.get("is_complete"),
            "validation_errors": validation.get("errors", []) or [],
            "validation_warnings": validation.get("warnings", []) or [],
            "rag_query": rag_query,
            "rag_results_count": len(rag_results),
            "vision_signature_detected": g(ocr, "ocr_data", "vision", "signature", "detected"),
            "vision_stamp_detected": g(ocr, "ocr_data", "vision", "stamp", "detected"),
            "language": g(ocr, "ocr_data", "language"),
            "page_count": g(ocr, "ocr_data", "page_count"),
        }

        return cls(
            document_text=document_text,
            document_type=document_type,
            summary=summary_text,
            intent=question,
            entities=[],
            topic=document_type,
            subtopics=[],
            requested_action=question,
            sender=extraction.get("sender"),
            recipient=None,
            institution=None,
            legal_references=[],
            previous_correspondence=[],
            attachments=[],
            metadata=metadata,
            classification_confidence=classification.get("classification_confidence"),
            analysis_confidence=None,
            writing_output=None,
        )


# --------------------------------------------------------------------------
# Knowledge base entities
# --------------------------------------------------------------------------

@dataclass
class Department:
    """One routable node in the institutional hierarchy."""

    institution: str
    department: str
    authority_level: str          # title of the responsible authority (role)
    responsibilities: List[str]
    handled_topics: List[str]
    keywords: List[str]
    presidency: Optional[str] = None      # Presidency / General Directorate
    unit: Optional[str] = None            # Directorate (alt Müdürlüğü)
    branch: Optional[str] = None          # Branch (Şube)
    parent_department: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    legal_authority: List[str] = field(default_factory=list)
    required_documents: List[str] = field(default_factory=list)
    excluded_topics: List[str] = field(default_factory=list)
    routing_rules: List[str] = field(default_factory=list)
    makam: Optional[str] = None           # concrete authority/makam name
    channel_hint: Optional[str] = None    # e.g. "LEGAL", "CONFIDENTIAL"

    @property
    def id(self) -> str:
        parts = [self.institution, self.department, self.unit or ""]
        return " / ".join(p for p in parts if p)

    def corpus_text(self) -> str:
        """Flattened text used for BM25 / semantic / keyword scoring."""
        parts = (
            self.responsibilities
            + self.handled_topics
            + self.keywords
            + self.entities
            + self.legal_authority
        )
        return " ".join(parts)


@dataclass
class IntentSegment:
    """One detected request/intent inside a (possibly multi-intent) document."""

    label: str
    text: str
    source_topic: Optional[str] = None


# --------------------------------------------------------------------------
# Scoring / candidates
# --------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    rule: float = 0.0
    keyword: float = 0.0
    bm25: float = 0.0
    semantic: float = 0.0
    metadata: float = 0.0
    legal: float = 0.0
    entity: float = 0.0
    llm: float = 0.0
    total: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "rule": round(self.rule, 4),
            "keyword": round(self.keyword, 4),
            "bm25": round(self.bm25, 4),
            "semantic": round(self.semantic, 4),
            "metadata": round(self.metadata, 4),
            "legal": round(self.legal, 4),
            "entity": round(self.entity, 4),
            "llm": round(self.llm, 4),
            "total": round(self.total, 4),
        }


@dataclass
class RouteCandidate:
    department: Department
    score: ScoreBreakdown
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    rejected: bool = False
    rejection_reason: Optional[str] = None
    intent_label: str = "primary"


@dataclass
class Route:
    institution: str
    department: str
    unit: Optional[str]
    presidency: Optional[str]
    branch: Optional[str]
    authority: str
    makam: Optional[str]
    channel: str
    reason: str
    evidence: List[str]
    score: float
    rejection_reason: Optional[str] = None  # populated for alternative routes

    def as_dict(self) -> Dict[str, Any]:
        return {
            "institution": self.institution,
            "presidency": self.presidency,
            "department": self.department,
            "unit": self.unit,
            "branch": self.branch,
            "authority": self.authority,
            "makam": self.makam,
            "channel": self.channel,
            "reason": self.reason,
            "evidence": self.evidence,
            "score": round(self.score, 4),
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class RoutingResult:
    routing_status: str
    primary_route: Optional[Route]
    secondary_routes: List[Route]
    alternative_routes: List[Route]
    recommended_department: Optional[str]
    recommended_authority: Optional[str]
    routing_reason: Optional[str]
    routing_evidence: List[str]
    confidence: str
    ambiguities: List[str]
    conflicts: List[str]
    missing_information: List[str]
    needs_human_review: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "routing_status": self.routing_status,
            "primary_route": self.primary_route.as_dict() if self.primary_route else None,
            "secondary_routes": [r.as_dict() for r in self.secondary_routes],
            "alternative_routes": [r.as_dict() for r in self.alternative_routes],
            "recommended_department": self.recommended_department,
            "recommended_authority": self.recommended_authority,
            "routing_reason": self.routing_reason,
            "routing_evidence": self.routing_evidence,
            "confidence": self.confidence,
            "ambiguities": self.ambiguities,
            "conflicts": self.conflicts,
            "missing_information": self.missing_information,
            "needs_human_review": self.needs_human_review,
        }

    def as_contract_dict(self) -> Dict[str, Any]:
        """Collapse the full internal result down to the pipeline's public
        routing-stage output contract:

            {"success": bool, "department": string}

        `success` is False whenever:
          * no department could be recommended at all (e.g. empty
            knowledge base / no candidates retrieved), or
          * post-hoc validation flagged the decision as internally
            inconsistent (ROUTING_CONFLICT), or
          * the chosen primary route carries a `rejection_reason` -- i.e.
            it was only picked because *something* had to be returned, not
            because real evidence supported it (e.g. fully empty document
            text and question).

        Everything else (ambiguity, upstream signal conflicts, low
        confidence) still yields a best-effort department with
        success=True, since a low-confidence route is still a routed
        decision, not a failed one.
        """
        department = self.recommended_department
        primary_unsupported = bool(self.primary_route and self.primary_route.rejection_reason)
        success = (
            bool(department)
            and self.routing_status != RoutingStatus.ROUTING_CONFLICT.value
            and not primary_unsupported
        )
        return {
            "success": success,
            "department": department if success and department else "",
        }
