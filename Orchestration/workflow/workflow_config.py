"""
workflow_config.py
-------------------
Typed workflow configuration, loaded from config.yaml with safe defaults.

config.yaml controls, per stage: whether its Agent is enabled/integrated,
retry count, timeout, and fallback policy - i.e. exactly the policies the
brief calls out ("enabled Agents, retries, fallbacks and timeouts").

If config.yaml is missing or empty, sane defaults are used (all pipeline
stages enabled: OCR → … → Writing).
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set

import yaml

from Orchestration.graph.graph_definition import DEFAULT_SEQUENCE, Stage

logger = logging.getLogger("Orchestration.workflow_config")

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class FallbackPolicy(str, enum.Enum):
    TERMINATE = "terminate"
    SKIP = "skip"
    FALLBACK_STAGE = "fallback_stage"


@dataclass(frozen=True)
class StageConfig:
    stage: Stage
    enabled: bool = False
    module: str = ""
    class_name: str = ""
    retries: int = 0
    timeout_seconds: int = 60
    fallback: FallbackPolicy = FallbackPolicy.SKIP
    fallback_stage: Optional[str] = None


@dataclass(frozen=True)
class SupervisorConfig:
    mode: str = "deterministic"  # deterministic | llm (reserved for future use)
    max_total_retries: int = 12


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    redact_content: bool = True


@dataclass(frozen=True)
class WorkflowConfig:
    default_sequence: list = field(default_factory=lambda: list(DEFAULT_SEQUENCE))
    stages: Dict[Stage, StageConfig] = field(default_factory=dict)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def stage(self, stage: Stage) -> StageConfig:
        return self.stages[stage]

    @property
    def optional_stages(self) -> Set[Stage]:
        return {Stage.RAG}


# ---------------------------------------------------------------------------
# Defaults: full pipeline OCR → … → Writing wired to real Agents.
# ---------------------------------------------------------------------------
def _default_stages() -> Dict[Stage, StageConfig]:
    return {
        Stage.OCR: StageConfig(
            stage=Stage.OCR,
            enabled=True,
            module="Agents.ocr_agent",
            class_name="OCRAgent",
            retries=1,
            timeout_seconds=240,
            fallback=FallbackPolicy.TERMINATE,
        ),
        Stage.CLASSIFICATION: StageConfig(
            stage=Stage.CLASSIFICATION,
            enabled=True,
            module="Agents.classification_agent",
            class_name="ClassificationAgent",
            retries=1,
            timeout_seconds=300,
            fallback=FallbackPolicy.SKIP,
        ),
        Stage.EXTRACTION: StageConfig(
            stage=Stage.EXTRACTION,
            enabled=True,
            module="Agents.extraction_agent",
            class_name="ExtractionAgent",
            retries=1,
            timeout_seconds=300,
            fallback=FallbackPolicy.SKIP,
        ),
        Stage.VALIDATION: StageConfig(
            stage=Stage.VALIDATION,
            enabled=True,
            module="Agents.validation_agent",
            class_name="ValidationAgent",
            retries=1,
            timeout_seconds=60,
            fallback=FallbackPolicy.SKIP,
        ),
        Stage.RAG: StageConfig(
            stage=Stage.RAG,
            enabled=True,
            module="Agents.rag_agent",
            class_name="RAGAgent",
            retries=1,
            timeout_seconds=120,
            fallback=FallbackPolicy.SKIP,
        ),
        Stage.SUMMARY: StageConfig(
            stage=Stage.SUMMARY,
            enabled=True,
            module="Agents.summary_agent",
            class_name="SummaryAgent",
            retries=1,
            timeout_seconds=120,
            fallback=FallbackPolicy.SKIP,
        ),
        Stage.ROUTING: StageConfig(
            stage=Stage.ROUTING,
            enabled=True,
            module="Agents.routing_agent",
            class_name="RoutingAgent",
            retries=1,
            timeout_seconds=30,
            fallback=FallbackPolicy.SKIP,
        ),
        Stage.WRITING: StageConfig(
            stage=Stage.WRITING,
            enabled=True,
            module="Agents.writer_agent",
            class_name="WriterAgent",
            retries=1,
            timeout_seconds=60,
            fallback=FallbackPolicy.SKIP,
        ),
    }


def _parse_stage(stage: Stage, raw: Dict[str, Any], default: StageConfig) -> StageConfig:
    fallback_raw = raw.get("fallback", default.fallback.value)
    try:
        fallback = FallbackPolicy(fallback_raw)
    except ValueError:
        logger.warning("invalid_fallback_policy stage=%s value=%s, using 'skip'", stage.value, fallback_raw)
        fallback = FallbackPolicy.SKIP

    return StageConfig(
        stage=stage,
        enabled=bool(raw.get("enabled", default.enabled)),
        module=str(raw.get("module", default.module)),
        class_name=str(raw.get("class_name", default.class_name)),
        retries=int(raw.get("retries", default.retries)),
        timeout_seconds=int(raw.get("timeout_seconds", default.timeout_seconds)),
        fallback=fallback,
        fallback_stage=raw.get("fallback_stage", default.fallback_stage),
    )


def load_config(path: Optional[Path] = None) -> WorkflowConfig:
    """Load config.yaml, falling back to defaults for any missing section.

    An empty or absent config.yaml is valid and results in the default
    configuration (full pipeline enabled).
    """

    path = path or _CONFIG_PATH
    raw: Dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
        except yaml.YAMLError as exc:
            logger.warning("config_parse_error path=%s error=%s, using defaults", path, exc)

    defaults = _default_stages()
    agents_raw = raw.get("agents", {}) if isinstance(raw.get("agents"), dict) else {}
    stages: Dict[Stage, StageConfig] = {}
    for stage, default_cfg in defaults.items():
        stage_raw = agents_raw.get(stage.value, {})
        stages[stage] = _parse_stage(stage, stage_raw if isinstance(stage_raw, dict) else {}, default_cfg)

    supervisor_raw = raw.get("supervisor", {}) if isinstance(raw.get("supervisor"), dict) else {}
    supervisor = SupervisorConfig(
        mode=str(supervisor_raw.get("mode", "deterministic")),
        max_total_retries=int(supervisor_raw.get("max_total_retries", 12)),
    )

    logging_raw = raw.get("logging", {}) if isinstance(raw.get("logging"), dict) else {}
    logging_cfg = LoggingConfig(
        level=str(logging_raw.get("level", "INFO")),
        redact_content=bool(logging_raw.get("redact_content", True)),
    )

    return WorkflowConfig(stages=stages, supervisor=supervisor, logging=logging_cfg)


__all__ = [
    "FallbackPolicy",
    "StageConfig",
    "SupervisorConfig",
    "LoggingConfig",
    "WorkflowConfig",
    "load_config",
]
