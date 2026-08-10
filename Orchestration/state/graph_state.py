"""
graph_state.py
----------------
Shared LangGraph state passed between Supervisor and every Agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    """Mutable workflow state. Agents merge their outputs into this dict."""

    document_id: str
    document_path: str
    document_text: str

    # Partial agent outputs
    ocr_result: Dict[str, Any]
    classification_result: Dict[str, Any]
    extraction_result: Dict[str, Any]
    validation_result: Dict[str, Any]
    rag_query: str
    rag_result: Dict[str, Any]
    summary: str
    draft_letter: str
    routing_decision: Dict[str, Any]

    # Control
    errors: List[str]
    current_agent: Optional[str]
    final_decision: Dict[str, Any]
