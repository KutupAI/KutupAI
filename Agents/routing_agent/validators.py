"""
validators.py
==============

Two validation passes:

  1. validate_shared_state  - runs BEFORE routing. Never hard-fails the
     pipeline (routing must still attempt a best-effort decision per the
     spec's "low confidence should not force a wrong decision" principle),
     but reports missing_information used later for confidence calibration.

  2. validate_routing_result - runs AFTER a decision is built. Confirms
     internal consistency (authority belongs to the chosen department,
     hierarchy fields are populated, multi-route/alternative shapes match
     the declared routing_status). Returns a list of problems; agent.py
     downgrades confidence / flags ROUTING_CONFLICT when this is non-empty.
"""

from __future__ import annotations

from typing import List

from .knowledge_base import KnowledgeBase
from .models import RoutingResult, RoutingStatus, SharedStateInput

REQUIRED_SOFT_FIELDS = [
    ("document_text", "document text"),
    ("topic", "topic"),
    ("intent", "intent"),
    ("entities", "entities"),
    ("sender", "sender"),
    ("recipient", "recipient"),
]


def validate_shared_state(state: SharedStateInput) -> List[str]:
    """Never raises for missing optional fields -- only document_text being
    completely empty is treated as a hard problem, reported (not raised) so
    the caller can decide how to handle it."""
    missing: List[str] = []

    if not state.document_text or not state.document_text.strip():
        missing.append("document_text is empty or missing")

    for field_name, label in REQUIRED_SOFT_FIELDS[1:]:
        value = getattr(state, field_name, None)
        if value in (None, "", [], {}):
            missing.append(f"{label} not provided by upstream agents")

    return missing


def validate_routing_result(result: RoutingResult, kb: KnowledgeBase) -> List[str]:
    problems: List[str] = []

    if result.routing_status in (RoutingStatus.SINGLE_ROUTE.value, RoutingStatus.MULTI_ROUTE.value,
                                  RoutingStatus.AMBIGUOUS.value):
        if result.primary_route is None:
            problems.append("primary_route missing for a non-conflict routing_status")
        else:
            dept = kb.get_by_name(result.primary_route.department)
            if dept is None:
                problems.append(f"primary_route department '{result.primary_route.department}' not found in knowledge base")
            else:
                if result.primary_route.authority != dept.authority_level:
                    problems.append(
                        f"authority mismatch: route has '{result.primary_route.authority}', "
                        f"knowledge base defines '{dept.authority_level}' for this department"
                    )
                if not result.primary_route.institution:
                    problems.append("primary_route missing institution in hierarchy")

    if result.routing_status == RoutingStatus.MULTI_ROUTE.value and not result.secondary_routes:
        problems.append("routing_status is MULTI_ROUTE but no secondary_routes were produced")

    if result.routing_status != RoutingStatus.MULTI_ROUTE.value and result.secondary_routes:
        problems.append("secondary_routes present but routing_status is not MULTI_ROUTE")

    if result.confidence not in ("HIGH", "MEDIUM", "LOW"):
        problems.append(f"invalid confidence level '{result.confidence}'")

    return problems
