"""
routing_logic.py
-----------------
Deterministic transition rules between Agents (e.g. after OCR ->
Classification, unless OCR failed).

This module is pure: given the current stage, the current GraphState and
the resolved StageConfig, it returns a Decision describing what the
Supervisor should do next. It has no side effects and does not mutate
state - that stays StateManager's job.

NOTE on naming: this is Orchestration routing ("which Agent runs next?").
It is intentionally distinct from the `routing_agent` business Agent
(Stage.ROUTING, "which department/business destination?"), whose result
just happens to be read here like any other stage's output.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

from Orchestration.graph.graph_definition import DEFAULT_SEQUENCE, Stage, stage_order_index
from Orchestration.messages.message_schema import ExecutionStatus
from Orchestration.state.graph_state import GraphState
from Orchestration.workflow.workflow_config import FallbackPolicy, StageConfig, WorkflowConfig


class Action(str, enum.Enum):
    CONTINUE = "continue"      # advance to the returned next_stage
    RETRY = "retry"            # re-run the current stage
    FALLBACK = "fallback"      # jump to a configured fallback stage
    SKIP = "skip"              # current stage not available; move on
    TERMINATE = "terminate"    # unrecoverable; stop the workflow
    COMPLETE = "complete"      # ran the last stage; workflow is done


@dataclass(frozen=True)
class Decision:
    action: Action
    next_stage: Optional[Stage]
    reason: str


def _next_in_sequence(stage: Stage) -> Optional[Stage]:
    idx = stage_order_index(stage)
    if idx + 1 < len(DEFAULT_SEQUENCE):
        return DEFAULT_SEQUENCE[idx + 1]
    return None


def decide_before_execution(stage: Stage, state: GraphState, config: WorkflowConfig, *, available: bool) -> Decision:
    """Decide whether `stage` should even run, before the Agent adapter is
    invoked. `available` reflects whether a real (or test-overridden) Agent
    adapter is actually registered for this stage - not just the raw
    config.yaml `enabled` flag, so `agent_overrides` used in tests can make
    a stage runnable even while it stays disabled by default in
    config.yaml."""

    if not available:
        stage_cfg = config.stage(stage)
        node_is_optional = stage in config.optional_stages
        reason = f"{stage.value} agent not available (not integrated/enabled)"
        if node_is_optional or stage_cfg.fallback == FallbackPolicy.SKIP:
            return Decision(Action.SKIP, _next_in_sequence(stage), reason)
        return Decision(Action.TERMINATE, None, reason)
    return Decision(Action.CONTINUE, stage, "stage available")


def decide_after_execution(
    stage: Stage,
    status: ExecutionStatus,
    state: GraphState,
    config: WorkflowConfig,
) -> Decision:
    """Decide what happens after one execution attempt of `stage`."""

    stage_cfg = config.stage(stage)

    if status == ExecutionStatus.SUCCESS:
        return _advance(stage, state, config)

    attempts_so_far = state["stage_retries"].get(stage.value, 0)
    if attempts_so_far < stage_cfg.retries:
        return Decision(
            Action.RETRY,
            stage,
            f"{stage.value} status={status.value}, retrying "
            f"({attempts_so_far + 1}/{stage_cfg.retries})",
        )

    # Retries exhausted (or none configured): apply the stage's fallback policy.
    if stage_cfg.fallback == FallbackPolicy.SKIP:
        return Decision(
            Action.SKIP,
            _next_in_sequence(stage),
            f"{stage.value} failed after retries, skipping (non-critical stage)",
        )
    if stage_cfg.fallback == FallbackPolicy.FALLBACK_STAGE and stage_cfg.fallback_stage:
        return Decision(
            Action.FALLBACK,
            Stage(stage_cfg.fallback_stage),
            f"{stage.value} failed after retries, falling back to {stage_cfg.fallback_stage}",
        )
    return Decision(
        Action.TERMINATE,
        None,
        f"{stage.value} failed after retries and policy is 'terminate'",
    )


def _advance(stage: Stage, state: GraphState, config: WorkflowConfig) -> Decision:
    """Business-level branching on top of the default linear sequence."""

    if stage == Stage.CLASSIFICATION:
        classification = state.get("classification") or state.get("classification_result") or {}
        if isinstance(classification, dict) and classification.get("requires_rag") is False:
            return Decision(Action.CONTINUE, Stage.EXTRACTION, "classification: proceeding to extraction")

    if stage == Stage.VALIDATION:
        validation = state.get("validation") or state.get("validation_result") or {}
        if isinstance(validation, dict) and validation.get("requires_rag") is False:
            return Decision(Action.CONTINUE, Stage.SUMMARY, "validation: RAG not required, skipping to summary")

    next_stage = _next_in_sequence(stage)
    if next_stage is None:
        return Decision(Action.COMPLETE, None, f"{stage.value} was the final stage")
    return Decision(Action.CONTINUE, next_stage, f"{stage.value} succeeded, advancing to {next_stage.value}")


__all__ = ["Action", "Decision", "decide_before_execution", "decide_after_execution"]
