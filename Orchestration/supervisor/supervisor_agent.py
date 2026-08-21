"""
supervisor_agent.py
--------------------
The central brain of the Orchestration layer: reads current state, decides
whether the current stage should run at all (via routing_logic), and after
each execution decides what happens next (continue / retry / fallback /
skip / terminate / complete).

The Supervisor never talks to Agents directly - that stays the Workflow's
job (workflow_builder.py). The Supervisor only makes decisions; execution
and state writes are delegated to the Workflow/StateManager.

`supervisor.mode` in config.yaml is reserved for a future LLM-assisted
Supervisor (see supervisor_prompts.py); the default and only implemented
mode today is "deterministic", driven entirely by routing_logic.py.
"""

from __future__ import annotations

import logging

from Orchestration.graph.graph_definition import DEFAULT_SEQUENCE, Stage
from Orchestration.messages.message_schema import ExecutionStatus
from Orchestration.state.graph_state import GraphState
from Orchestration.supervisor.routing_logic import (
    Action,
    Decision,
    decide_after_execution,
    decide_before_execution,
)
from Orchestration.workflow.workflow_config import WorkflowConfig

logger = logging.getLogger("Orchestration.supervisor")


class SupervisorAgent:
    """Workflow controller. Stateless aside from its config."""

    def __init__(self, config: WorkflowConfig) -> None:
        self.config = config
        if self.config.supervisor.mode != "deterministic":
            logger.warning(
                "supervisor_mode=%s not implemented, falling back to deterministic",
                self.config.supervisor.mode,
            )

    def initial_stage(self) -> Stage:
        return DEFAULT_SEQUENCE[0]

    def decide_before(self, stage: Stage, state: GraphState, *, available: bool) -> Decision:
        decision = decide_before_execution(stage, state, self.config, available=available)
        logger.info(
            "supervisor_decision workflow_id=%s stage=%s phase=before action=%s reason=%s",
            state.get("workflow_id"),
            stage.value,
            decision.action.value,
            decision.reason,
        )
        return decision

    def decide_after(self, stage: Stage, status: ExecutionStatus, state: GraphState) -> Decision:
        decision = decide_after_execution(stage, status, state, self.config)
        logger.info(
            "supervisor_decision workflow_id=%s stage=%s phase=after status=%s action=%s reason=%s",
            state.get("workflow_id"),
            stage.value,
            status.value,
            decision.action.value,
            decision.reason,
        )
        return decision


__all__ = ["SupervisorAgent", "Decision", "Action"]
