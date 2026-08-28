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


def _empty_rag_slot(*, reason: str, code: str = "rag_unavailable") -> Dict[str, Any]:
    return {
        "success": False,
        "rag_data": {"operation": "retrieve", "query": "", "results": []},
        "error": {"code": code, "message": reason},
    }


def _apply_rag_slot(state: GraphState, rag_slot: Dict[str, Any]) -> None:
    state["rag"] = rag_slot  # type: ignore[literal-required]
    state["rag_result"] = {  # type: ignore[literal-required]
        "success": bool(rag_slot.get("success")),
        "data": rag_slot.get("rag_data"),
        "error": rag_slot.get("error"),
    }
    state["rag_status"] = "completed" if rag_slot.get("success") else "failed"


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

    if not isinstance(updated_state, dict):
        return None

    # Prefer the unified short-key contract when the Agent wrote a non-empty
    # section (seeded empty {} from StateManager must not win over *_result).
    if stage == Stage.OCR and isinstance(updated_state.get("ocr"), dict) and updated_state["ocr"]:
        return updated_state["ocr"]
    if (
        stage == Stage.CLASSIFICATION
        and isinstance(updated_state.get("classification"), dict)
        and updated_state["classification"]
    ):
        return updated_state["classification"]
    if (
        stage == Stage.EXTRACTION
        and isinstance(updated_state.get("extraction"), dict)
        and updated_state["extraction"]
    ):
        return updated_state["extraction"]
    if (
        stage == Stage.VALIDATION
        and isinstance(updated_state.get("validation"), dict)
        and updated_state["validation"]
    ):
        return updated_state["validation"]
    if stage == Stage.RAG and isinstance(updated_state.get("rag"), dict) and updated_state["rag"]:
        return updated_state["rag"]
    if (
        stage == Stage.SUMMARY
        and isinstance(updated_state.get("summary"), dict)
        and updated_state["summary"]
    ):
        return updated_state["summary"]
    if stage == Stage.WRITING:
        if isinstance(updated_state.get("writing"), dict) and updated_state["writing"]:
            return updated_state["writing"]
        if "draft_letter" in updated_state:
            return updated_state["draft_letter"]
    if stage == Stage.ROUTING:
        if isinstance(updated_state.get("routing"), dict) and updated_state["routing"]:
            return updated_state["routing"]
        if "routing_decision" in updated_state:
            return updated_state["routing_decision"]

    result_key = f"{stage.value}_result"
    if result_key in updated_state:
        return updated_state[result_key]

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
            if stage == Stage.RAG:
                return AgentResult.fail(
                    stage.value,
                    ExecutionStatus.EXCEPTION,
                    str(exc),
                    error_type=type(exc).__name__,
                    data=_empty_rag_slot(reason=str(exc), code="rag_exception"),
                )
            return AgentResult.fail(
                stage.value, ExecutionStatus.EXCEPTION, str(exc), error_type=type(exc).__name__
            )

        if not isinstance(updated, dict):
            if stage == Stage.RAG:
                return AgentResult.fail(
                    stage.value,
                    ExecutionStatus.INVALID_RESULT,
                    f"agent returned {type(updated).__name__}, expected dict",
                    data=_empty_rag_slot(reason="invalid RAG agent return type", code="invalid_result"),
                )
            return AgentResult.fail(
                stage.value,
                ExecutionStatus.INVALID_RESULT,
                f"agent returned {type(updated).__name__}, expected dict",
            )

        status_key = f"{stage.value}_status"
        reported_status = updated.get(status_key)
        data = _extract_result_data(stage, updated, before)

        # RAG soft payload: apply and continue when the agent returned a NEW
        # structured rag section. Do not treat a leftover state["rag"] from a
        # previous attempt as success (retries must still observe status=failed).
        rag_newly_written = (
            stage == Stage.RAG
            and isinstance(updated.get("rag"), dict)
            and updated.get("rag")
            and updated.get("rag") is not before.get("rag")
            and updated.get("rag") != before.get("rag")
        )
        if rag_newly_written and (
            "rag_data" in updated["rag"] or "success" in updated["rag"]
        ):
            return AgentResult.ok(stage.value, updated["rag"])
        if stage == Stage.RAG and reported_status in ("failed", "error"):
            return AgentResult.fail(
                stage.value,
                ExecutionStatus.FAILURE,
                f"{stage.value} reported status={reported_status}",
                data=_empty_rag_slot(
                    reason=f"RAG reported status={reported_status}",
                    code="rag_status_failed",
                ),
            )
        if stage == Stage.RAG and isinstance(data, dict) and (
            "rag_data" in data or "success" in data
        ):
            return AgentResult.ok(stage.value, data)
        if stage == Stage.RAG:
            return AgentResult.ok(
                stage.value,
                _empty_rag_slot(reason="RAG agent returned no rag section", code="empty_rag"),
            )

        if reported_status in ("failed", "error"):
            return AgentResult.fail(stage.value, ExecutionStatus.FAILURE, f"{stage.value} reported status={reported_status}")
        if data is None:
            return AgentResult.fail(
                stage.value,
                ExecutionStatus.MISSING_STATE,
                f"{stage.value} agent returned no result section",
            )
        if stage == Stage.SUMMARY and isinstance(data, dict) and data.get("success") is False:
            return AgentResult.fail(
                stage.value, ExecutionStatus.FAILURE, "summary agent returned success=false"
            )
        if stage == Stage.ROUTING and isinstance(data, dict) and data.get("success") is False:
            return AgentResult.fail(
                stage.value, ExecutionStatus.FAILURE, "routing agent returned success=false"
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


class _LazyAgentAdapter:
    """Import/instantiate the real Agent on first use (not at registry build)."""

    not_integrated = False

    def __init__(self, stage: Stage, stage_cfg: StageConfig) -> None:
        self._stage = stage
        self._stage_cfg = stage_cfg
        self._adapter: Optional[AgentAdapter] = None
        self._load_error: Optional[str] = None

    def __call__(self, stage: Stage, state: GraphState, stage_cfg: StageConfig) -> AgentResult:
        if self._adapter is None and self._load_error is None:
            try:
                agent = _instantiate(self._stage_cfg)
                self._adapter = _default_adapter(agent)
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
                logger.warning(
                    "agent_not_available stage=%s module=%s class=%s error=%s",
                    self._stage.value,
                    self._stage_cfg.module,
                    self._stage_cfg.class_name,
                    exc,
                )
        if self._load_error is not None:
            return AgentResult.fail(
                self._stage.value,
                ExecutionStatus.NOT_INTEGRATED,
                f"{self._stage.value} agent could not be loaded: {self._load_error}",
            )
        assert self._adapter is not None
        return self._adapter(stage, state, stage_cfg)


def build_agent_registry(
    config: WorkflowConfig,
    agent_overrides: Optional[Dict[Stage, AgentProtocol]] = None,
) -> Dict[Stage, AgentAdapter]:
    """Resolve one adapter per stage.

    Precedence: explicit override (tests / gradual rollout) > configured
    real Agent (lazily imported on first stage run) > not-integrated placeholder.
    """

    agent_overrides = agent_overrides or {}
    registry: Dict[Stage, AgentAdapter] = {}
    for stage, stage_cfg in config.stages.items():
        if stage in agent_overrides:
            registry[stage] = _default_adapter(agent_overrides[stage])
            continue
        if not stage_cfg.enabled:
            registry[stage] = _not_integrated_adapter(
                stage, f"{stage.value} agent not enabled in config.yaml"
            )
            continue
        # Defer import/construction until the stage actually runs so enabling
        # Classification does not force OCR/torch to load at registry build.
        registry[stage] = _LazyAgentAdapter(stage, stage_cfg)
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

            # OCR sonucu aynı dosya için Orchestration cache'inden geldiyse
            # agent'i tekrar çalıştırma; normal başarı geçişini uygula.
            if stage == Stage.OCR and state.get("ocr_cache_hit") and state.get("ocr"):
                logger.info("ocr_cache_hit workflow_id=%s", state.get("workflow_id"))
                cached_result = AgentResult.ok(stage.value, state["ocr"])
                self.state_manager.apply_result(state, stage, cached_result, attempt=0)
                after_decision = self.supervisor.decide_after(stage, cached_result.status, state)
                if after_decision.action == Action.COMPLETE:
                    stage = None
                    break
                stage = after_decision.next_stage
                continue

            adapter = self.registry.get(stage)
            is_available = adapter is not None and not getattr(adapter, "not_integrated", False)

            before_decision = self.supervisor.decide_before(stage, state, available=is_available)
            if before_decision.action == Action.SKIP:
                state["stage_status"][stage.value] = ExecutionStatus.SKIPPED.value
                # Never leave Presentation with rag:{} when the stage is skipped.
                if stage == Stage.RAG and not (isinstance(state.get("rag"), dict) and state["rag"]):
                    _apply_rag_slot(
                        state,
                        _empty_rag_slot(
                            reason=before_decision.reason or "RAG stage skipped",
                            code="rag_skipped",
                        ),
                    )
                stage = before_decision.next_stage
                continue
            if before_decision.action == Action.TERMINATE:
                self.state_manager.terminate(state, before_decision.reason)
                break

            attempt = state["stage_retries"].get(stage.value, 0) + 1
            self.state_manager.begin_attempt(state, stage, attempt)

            if adapter is None:
                result = AgentResult.fail(
                    stage.value,
                    ExecutionStatus.NOT_INTEGRATED,
                    f"no adapter registered for {stage.value}",
                    data=_empty_rag_slot(reason="no RAG adapter", code="not_integrated")
                    if stage == Stage.RAG
                    else None,
                )
            else:
                started = time.monotonic()
                result = adapter(stage, state, self.config.stage(stage))
                elapsed = time.monotonic() - started
                timings = state.setdefault("stage_timings_ms", {})  # type: ignore[literal-required]
                timings[stage.value] = round(timings.get(stage.value, 0.0) + elapsed * 1000, 1)
                logger.info(
                    "stage_done stage=%s attempt=%s status=%s elapsed=%.2fs",
                    stage.value,
                    attempt,
                    getattr(result.status, "value", result.status),
                    elapsed,
                )
                timeout = self.config.stage(stage).timeout_seconds
                if elapsed > timeout:
                    logger.warning(
                        "agent_timeout_exceeded stage=%s elapsed=%.2fs timeout=%ss",
                        stage.value,
                        elapsed,
                        timeout,
                    )

            self.state_manager.apply_result(state, stage, result, attempt)

            # Guarantee a rag slot even if apply_result could not merge data.
            if stage == Stage.RAG and not (isinstance(state.get("rag"), dict) and state["rag"]):
                slot = result.data if isinstance(result.data, dict) and result.data else None
                _apply_rag_slot(
                    state,
                    slot
                    if isinstance(slot, dict) and ("rag_data" in slot or "success" in slot)
                    else _empty_rag_slot(
                        reason=result.error or "RAG produced no payload",
                        code="rag_empty_after_apply",
                    ),
                )

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
