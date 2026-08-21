"""
graph_definition.py
--------------------
Declarative workflow topology: nodes (Stages) and the default linear
transitions between them.

This module is intentionally framework-free (no LangGraph / heavy graph
library dependency) so the Orchestration layer has no unnecessary runtime
dependencies. `workflow/workflow_builder.py` compiles this topology, together
with `supervisor/routing_logic.py` (for conditional edges), into an
executable workflow.

Only the topology lives here. "Which Agent implements a stage" is an
adapter-registry concern (workflow_builder.py); "should we branch/retry/
fallback right now" is a routing concern (supervisor/routing_logic.py).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class Stage(str, enum.Enum):
    """A node in the Orchestration workflow graph. One Stage == one Agent."""

    OCR = "ocr"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    VALIDATION = "validation"
    RAG = "rag"
    SUMMARY = "summary"
    ROUTING = "routing"
    WRITING = "writing"


END = "END"
"""Sentinel returned by the Supervisor/routing logic to signal a normal
(non-error) completion of the workflow."""

TERMINATED = "TERMINATED"
"""Sentinel returned when the workflow must stop early due to an
unrecoverable failure (exhausted retries with a 'terminate' fallback,
missing required state, etc.)."""


@dataclass(frozen=True)
class GraphNode:
    """Static description of one workflow node."""

    stage: Stage
    description: str
    required_state_keys: Tuple[str, ...] = ()
    optional: bool = False
    """If True, the default routing logic is allowed to skip this stage
    when its Agent is disabled/not yet integrated, without treating that
    as a failure."""


# ---------------------------------------------------------------------------
# Default topology: Orchestration's canonical 8-stage pipeline.
#
#   OCR -> Classification -> Extraction -> Validation -> RAG -> Summary
#        -> Routing -> Writing -> END
# ---------------------------------------------------------------------------
NODES: Dict[Stage, GraphNode] = {
    Stage.OCR: GraphNode(
        stage=Stage.OCR,
        description="Extract raw text/structure from the source document.",
        required_state_keys=("document_id",),
    ),
    Stage.CLASSIFICATION: GraphNode(
        stage=Stage.CLASSIFICATION,
        description="Classify the document/request type.",
        required_state_keys=("ocr_result",),
    ),
    Stage.EXTRACTION: GraphNode(
        stage=Stage.EXTRACTION,
        description="Extract structured fields relevant to the classification.",
        required_state_keys=("ocr_result",),
    ),
    Stage.VALIDATION: GraphNode(
        stage=Stage.VALIDATION,
        description="Validate extracted fields against business rules.",
        required_state_keys=("extraction_result",),
    ),
    Stage.RAG: GraphNode(
        stage=Stage.RAG,
        description="Retrieve supporting knowledge/context when needed.",
        required_state_keys=(),
        optional=True,
    ),
    Stage.SUMMARY: GraphNode(
        stage=Stage.SUMMARY,
        description="Summarize the processed document/case.",
        required_state_keys=(),
    ),
    Stage.ROUTING: GraphNode(
        stage=Stage.ROUTING,
        description="Decide the destination department/business route "
        "(routing_agent - distinct from Orchestration's own agent-to-agent "
        "routing, see supervisor/routing_logic.py).",
        required_state_keys=(),
    ),
    Stage.WRITING: GraphNode(
        stage=Stage.WRITING,
        description="Produce the final written artifact (letter/response).",
        required_state_keys=(),
    ),
}

DEFAULT_SEQUENCE: List[Stage] = [
    Stage.OCR,
    Stage.CLASSIFICATION,
    Stage.EXTRACTION,
    Stage.VALIDATION,
    Stage.RAG,
    Stage.SUMMARY,
    Stage.ROUTING,
    Stage.WRITING,
]
"""Default, linear stage order. Actual transitions at runtime are decided by
the Supervisor (supervisor_agent.py + routing_logic.py), which may skip,
retry, fall back to a different stage, or terminate early."""


def default_edges() -> Dict[Stage, Optional[Stage]]:
    """Static (unconditional) linear edges: stage -> next stage, or None at
    the end of the sequence. This is the graph's "happy path" topology;
    the Supervisor may still deviate from it based on runtime state."""

    edges: Dict[Stage, Optional[Stage]] = {}
    for i, stage in enumerate(DEFAULT_SEQUENCE):
        edges[stage] = DEFAULT_SEQUENCE[i + 1] if i + 1 < len(DEFAULT_SEQUENCE) else None
    return edges


def stage_order_index(stage: Stage) -> int:
    return DEFAULT_SEQUENCE.index(stage)


__all__ = [
    "Stage",
    "GraphNode",
    "NODES",
    "DEFAULT_SEQUENCE",
    "END",
    "TERMINATED",
    "default_edges",
    "stage_order_index",
]
