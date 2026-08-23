"""
workflow_builder.py
--------------------
Builds the full executable workflow: which node (Stage) maps to which
Agent, and the run-loop that drives Orchestration -> Agent -> Result ->
State Update -> Decision -> Next Agent.

Agent integration contract
---------------------------
An Agent is any object exposing:

    def run(self, state: dict) -> dict: ...

(this is the exact interface `Agents.ocr_agent.OCRAgent` already
implements). The adapter below calls `agent.run(dict(state))` and reads
back one of two shapes from the returned dict:

  * `state["<stage>_result"]` / `state["<stage>_status"]` populated
    (preferred, matches the existing OCR contract), or
  * a plain dict considered the stage's result as-is.

Agents are located by dotted `module` + `class_name` from config.yaml and
imported lazily (only when their stage actually runs), so Orchestration
stays importable even before every Agent exists yet. Import/instantiation
failures surface as ExecutionStatus.NOT_INTEGRATED, never as a fabricated
successful result - Orchestration does not create fake Agent
implementations.

For tests, real Agents can be swapped for mocks via `agent_overrides`
(stage -> Agent-shaped instance) without touching config.yaml or real
Agent code.
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from Orchestration.graph.graph_definition import Stage
from Orchestration.messages.message_schema import AgentResult, ExecutionStatus
from Orchestration.state.graph_state import GraphState
from Orchestration.state.state_manager import StateManager
from Orchestration.supervisor.routing_logic import Action
from Orchestration.supervisor.supervisor_agent import SupervisorAgent
from Orchestration.workflow.workflow_config import StageConfig, WorkflowConfig, load_config

logger = logging.getLogger("Orchestration.workflow_builder")


@runtime_checkable
class AgentProtocol(Protocol):
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]: ...


AgentAdapter = Callable[[Stage, GraphState, StageConfig], AgentResult]
"""An adapter turns (stage, state, stage_config) into an AgentResult. The
default adapter (`_default_adapter`) wraps any AgentProtocol-shaped Agent;
callers may supply a fully custom adapter per stage if an Agent needs
bespoke translation."""


def _import_agent(stage_cfg: StageConfig) -> Any:
    module = importlib.import_module(stage_cfg.module)
    return getattr(module, stage_cfg.class_name)


def _instantiate(stage_cfg: StageConfig) -> AgentProtocol:
    agent_cls = _import_agent(stage_cfg)
    return agent_cls()  # Agents are expected to be zero-arg constructible.


def _extract_result_data(stage: Stage, updated_state: Dict[str, Any], before: Dict[str, Any]) -> Optional[Any]:
    """Best-effort extraction of the stage's own output section from
    whatever the Agent returned, without ever reading/writing another
    stage's section."""

    result_key = f"{stage.value}_result"
    if isinstance(updated_state, dict) and result_key in updated_state:
        return updated_state[result_key]
    if stage == Stage.SUMMARY and isinstance(updated_state, dict) and "summary" in updated_state:
        return updated_state["summary"]
    if stage == Stage.WRITING and isinstance(updated_state, dict):
        if "writing" in updated_state:
            return updated_state["writing"]
        # Supports older agent implementations during the migration.
        if "draft_letter" in updated_state:
            return updated_state["draft_letter"]
    if stage == Stage.ROUTING and isinstance(updated_state, dict) and "routing_decision" in updated_state:
        return updated_state["routing_decision"]
    # Fall back to "whatever new top-level keys the agent added" so a
    # differently-shaped (but well-behaved) Agent still produces a result
    # instead of silently losing its output.
    diff = {k: v for k, v in updated_state.items() if before.get(k) is not v and k not in before}
    return diff or None


def _default_adapter(agent: AgentProtocol) -> AgentAdapter:
    def adapter(stage: Stage, state: GraphState, stage_cfg: StageConfig) -> AgentResult:
        before = dict(state)
        try:
            updated = agent.run(dict(state))
        except Exception as exc:  # noqa: BLE001 - Agent internals are opaque here
            logger.exception("agent_exception stage=%s", stage.value)
            return AgentResult.fail(
                stage.value, ExecutionStatus.EXCEPTION, str(exc), error_type=type(exc).__name__
            )

        if not isinstance(updated, dict):
            return AgentResult.fail(
                stage.value,
                ExecutionStatus.INVALID_RESULT,
                f"agent returned {type(updated).__name__}, expected dict",
            )

        status_key = f"{stage.value}_status"
        reported_status = updated.get(status_key)
        data = _extract_result_data(stage, updated, before)

        if reported_status in ("failed", "error"):
            return AgentResult.fail(stage.value, ExecutionStatus.FAILURE, f"{stage.value} reported status={reported_status}")
        if data is None:
            return AgentResult.fail(
                stage.value, ExecutionStatus.MISSING_STATE, f"{stage.value} agent returned no result section"
            )
        if stage == Stage.SUMMARY and isinstance(data, dict) and data.get("success") is False:
            return AgentResult.fail(
                stage.value, ExecutionStatus.FAILURE, "summary agent returned success=false"
            )
        if stage == Stage.WRITING and isinstance(data, dict) and data.get("success") is False:
            return AgentResult.fail(
                stage.value, ExecutionStatus.FAILURE, "writer agent returned success=false"
            )
        return AgentResult.ok(stage.value, data)

    return adapter


class _NotIntegratedAdapter:
    """Callable adapter used when a stage has no real (or overridden) Agent
    yet. Marked with `not_integrated = True` so the Workflow can decide to
    skip/terminate *before* even attempting execution, without relying on
    raw config flags (which would ignore test-time `agent_overrides`)."""

    not_integrated = True

    def __init__(self, stage: Stage, reason: str) -> None:
        self._stage = stage
        self._reason = reason

    def __call__(self, stage: Stage, state: GraphState, stage_cfg: StageConfig) -> AgentResult:
        return AgentResult.fail(self._stage.value, ExecutionStatus.NOT_INTEGRATED, self._reason)


def _not_integrated_adapter(stage: Stage, reason: str) -> AgentAdapter:
    return _NotIntegratedAdapter(stage, reason)


def build_agent_registry(
    config: WorkflowConfig,
    agent_overrides: Optional[Dict[Stage, AgentProtocol]] = None,
) -> Dict[Stage, AgentAdapter]:
    """Resolve one adapter per stage.

    Precedence: explicit override (tests / gradual rollout) > configured
    real Agent (lazily imported) > not-integrated placeholder.
    """

    agent_overrides = agent_overrides or {}
    registry: Dict[Stage, AgentAdapter] = {}
    for stage, stage_cfg in config.stages.items():
        if stage in agent_overrides:
            registry[stage] = _default_adapter(agent_overrides[stage])
            continue
        if not stage_cfg.enabled:
            registry[stage] = _not_integrated_adapter(stage, f"{stage.value} agent not enabled in config.yaml")
            continue
        try:
            agent = _instantiate(stage_cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "agent_not_available stage=%s module=%s class=%s error=%s",
                stage.value,
                stage_cfg.module,
                stage_cfg.class_name,
                exc,
            )
            registry[stage] = _not_integrated_adapter(
                stage, f"{stage.value} agent could not be loaded: {exc}"
            )
            continue
        registry[stage] = _default_adapter(agent)
    return registry


@dataclass
class WorkflowResult:
    state: GraphState
    completed: bool
    terminated: bool


class Workflow:
    """Executable workflow: drives the Orchestration -> Agent -> Result ->
    State Update -> Decision -> Next Agent loop until completion or
    termination."""

    def __init__(
        self,
        config: WorkflowConfig,
        registry: Dict[Stage, AgentAdapter],
        state_manager: Optional[StateManager] = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.state_manager = state_manager or StateManager()
        self.supervisor = SupervisorAgent(config)

    def run(self, request: Dict[str, Any]) -> WorkflowResult:
        state = self.state_manager.initialize(request)
        stage: Optional[Stage] = self.supervisor.initial_stage()
        total_attempts = 0

        while stage is not None:
            total_attempts += 1
            if total_attempts > self.config.supervisor.max_total_retries + len(self.config.stages):
                self.state_manager.terminate(state, "max_total_retries exceeded")
                break

            adapter = self.registry.get(stage)
            is_available = adapter is not None and not getattr(adapter, "not_integrated", False)

            before_decision = self.supervisor.decide_before(stage, state, available=is_available)
            if before_decision.action == Action.SKIP:
                state["stage_status"][stage.value] = ExecutionStatus.SKIPPED.value
                stage = before_decision.next_stage
                continue
            if before_decision.action == Action.TERMINATE:
                self.state_manager.terminate(state, before_decision.reason)
                break

            attempt = state["stage_retries"].get(stage.value, 0) + 1
            self.state_manager.begin_attempt(state, stage, attempt)

            if adapter is None:
                result = AgentResult.fail(stage.value, ExecutionStatus.NOT_INTEGRATED, f"no adapter registered for {stage.value}")
            else:
                started = time.monotonic()
                result = adapter(stage, state, self.config.stage(stage))
                elapsed = time.monotonic() - started
                timeout = self.config.stage(stage).timeout_seconds
                if elapsed > timeout:
                    logger.warning(
                        "agent_timeout_exceeded stage=%s elapsed=%.2fs timeout=%ss",
                        stage.value,
                        elapsed,
                        timeout,
                    )

            self.state_manager.apply_result(state, stage, result, attempt)

            after_decision = self.supervisor.decide_after(stage, result.status, state)
            if after_decision.action == Action.RETRY:
                self.state_manager.increment_retry(state, stage)
                continue  # re-run same stage
            if after_decision.action == Action.TERMINATE:
                self.state_manager.terminate(state, after_decision.reason)
                break
            if after_decision.action == Action.COMPLETE:
                stage = None
                break
            # CONTINUE / FALLBACK / SKIP all move to another stage.
            stage = after_decision.next_stage

        terminated = bool(state.get("terminated"))
        completed = not terminated
        final_decision = {
            "completed": completed,
            "terminated": terminated,
            "termination_reason": state.get("termination_reason"),
            "final_stage_status": dict(state.get("stage_status", {})),
        }
        self.state_manager.finalize(state, final_decision)
        return WorkflowResult(state=state, completed=completed, terminated=terminated)


def build_workflow(
    config: Optional[WorkflowConfig] = None,
    agent_overrides: Optional[Dict[Stage, AgentProtocol]] = None,
    state_manager: Optional[StateManager] = None,
) -> Workflow:
    """Application-facing factory: construct a ready-to-run Workflow."""

    cfg = config or load_config()
    registry = build_agent_registry(cfg, agent_overrides)
    return Workflow(cfg, registry, state_manager=state_manager)


__all__ = [
    "AgentProtocol",
    "AgentAdapter",
    "WorkflowResult",
    "Workflow",
    "build_agent_registry",
    "build_workflow",
]
