from Orchestration.graph.graph_definition import Stage
from Orchestration.messages.message_schema import ExecutionStatus
from Orchestration.workflow.workflow_builder import build_agent_registry, build_workflow
from Orchestration.workflow.workflow_config import FallbackPolicy, load_config

from Orchestration.tests.mock_agents import (
    AlwaysFailingAgent,
    ExceptionAgent,
    InvalidResultAgent,
    MockClassificationAgent,
    MockExtractionAgent,
    MockOCRAgent,
    MockRagAgent,
    MockRoutingAgent,
    MockSummaryAgent,
    MockValidationAgent,
    MockWriterAgent,
)


def _full_overrides(**kwargs):
    overrides = {
        Stage.OCR: MockOCRAgent(),
        Stage.CLASSIFICATION: MockClassificationAgent(),
        Stage.EXTRACTION: MockExtractionAgent(),
        Stage.VALIDATION: MockValidationAgent(),
        Stage.RAG: MockRagAgent(),
        Stage.SUMMARY: MockSummaryAgent(),
        Stage.ROUTING: MockRoutingAgent(),
        Stage.WRITING: MockWriterAgent(),
    }
    overrides.update(kwargs)
    return overrides


def test_full_happy_path_runs_every_stage_in_order():
    wf = build_workflow(agent_overrides=_full_overrides())
    result = wf.run({"document_id": "doc-1", "document_path": "/tmp/a.pdf"})

    assert result.completed is True
    assert result.terminated is False
    stages_run = [h["stage"] for h in result.state["history"]]
    assert stages_run == [
        "ocr",
        "classification",
        "extraction",
        "validation",
        "rag",
        "summary",
        "routing",
        "writing",
    ]
    assert all(h["status"] == "success" for h in result.state["history"])
    assert result.state["writing"] == {"success": True, "answer": "Dear Sir, ..."}
    assert result.state["final_decision"]["completed"] is True


def test_default_config_runs_full_pipeline_with_mocks():
    # All stages enabled in config.yaml; mock Agents avoid live Inference.
    wf = build_workflow(agent_overrides=_full_overrides())
    result = wf.run({"document_id": "doc-1", "document_path": "/tmp/a.pdf"})

    assert result.completed is True
    statuses = result.state["stage_status"]
    assert statuses[Stage.OCR.value] == "success"
    assert statuses[Stage.CLASSIFICATION.value] == "success"
    assert statuses[Stage.EXTRACTION.value] == "success"
    assert statuses[Stage.VALIDATION.value] == "success"
    assert statuses[Stage.RAG.value] == "success"
    assert statuses[Stage.SUMMARY.value] == "success"
    assert statuses[Stage.ROUTING.value] == "success"
    assert statuses[Stage.WRITING.value] == "success"
    assert result.state["classification"] == {
        "success": True,
        "document_type": "invoice",
        "classification_confidence": 0.95,
    }
    assert result.state["extraction"] == {
        "success": True,
        "sender": None,
        "date": None,
        "address": None,
        "phone": None,
        "email": None,
    }
    assert "success" in result.state["validation_result"]
    assert "is_complete" in result.state["validation_result"]
    assert result.state["rag"]["success"] is True
    assert result.state["rag"]["rag_data"]["operation"] == "retrieve"
    assert result.state["rag_result"]["data"] == result.state["rag"]["rag_data"]
    # Unified OCR slot must be filled (not left as empty {}).
    assert isinstance(result.state.get("ocr"), dict) and result.state["ocr"]
    assert result.state["ocr"].get("success") is True
    assert "ocr_data" in result.state["ocr"]


def test_rag_optional_skip_after_retries_exhausted_does_not_terminate():
    failing_rag = AlwaysFailingAgent("rag")
    overrides = _full_overrides(**{Stage.RAG: failing_rag})
    wf = build_workflow(agent_overrides=overrides)
    result = wf.run({"document_id": "doc-1", "document_path": "/tmp/a.pdf"})

    assert result.completed is True
    assert result.terminated is False
    # rag has retries=1 -> 2 attempts total before falling back (skip)
    assert failing_rag.calls == 2
    assert result.state["stage_status"][Stage.RAG.value] == "failure"
    # workflow still reached the final stage
    assert result.state["stage_status"][Stage.WRITING.value] == "success"


def test_required_stage_failure_terminates_workflow():
    failing_ocr = AlwaysFailingAgent("ocr")
    wf = build_workflow(agent_overrides={Stage.OCR: failing_ocr})
    result = wf.run({"document_id": "doc-1", "document_path": "/tmp/a.pdf"})

    assert result.completed is False
    assert result.terminated is True
    assert result.state["termination_reason"] is not None
    # ocr has retries=1 -> 2 attempts total before terminating
    assert failing_ocr.calls == 2


def test_agent_exception_is_captured_not_raised():
    exploding_ocr = ExceptionAgent("ocr")
    wf = build_workflow(agent_overrides={Stage.OCR: exploding_ocr})
    result = wf.run({"document_id": "doc-1", "document_path": "/tmp/a.pdf"})

    assert result.terminated is True
    assert result.state["stage_status"][Stage.OCR.value] == ExecutionStatus.EXCEPTION.value
    assert result.state["errors"][-1]["error_type"] == "RuntimeError"


def test_invalid_agent_result_is_handled():
    wf = build_workflow(agent_overrides={Stage.OCR: InvalidResultAgent()})
    result = wf.run({"document_id": "doc-1", "document_path": "/tmp/a.pdf"})

    assert result.terminated is True
    assert result.state["stage_status"][Stage.OCR.value] == ExecutionStatus.INVALID_RESULT.value


def test_agent_override_takes_precedence_over_config_disabled_flag():
    # Force-disable summary in a local config copy; override should still run.
    config = load_config()
    summary_cfg = config.stage(Stage.SUMMARY)
    from dataclasses import replace

    disabled_summary = replace(summary_cfg, enabled=False)
    stages = dict(config.stages)
    stages[Stage.SUMMARY] = disabled_summary
    config = replace(config, stages=stages)
    assert config.stage(Stage.SUMMARY).enabled is False

    registry = build_agent_registry(config, {Stage.SUMMARY: MockSummaryAgent()})
    adapter = registry[Stage.SUMMARY]
    assert not getattr(adapter, "not_integrated", False)


def test_not_integrated_stage_has_marker_and_never_fabricates_success():
    # Disabled stages get a not_integrated placeholder (never a fake success).
    config = load_config()
    from dataclasses import replace

    summary_cfg = replace(config.stage(Stage.SUMMARY), enabled=False)
    stages = dict(config.stages)
    stages[Stage.SUMMARY] = summary_cfg
    config = replace(config, stages=stages)

    registry = build_agent_registry(config, {})
    adapter = registry[Stage.SUMMARY]
    assert getattr(adapter, "not_integrated", False) is True
    result = adapter(Stage.SUMMARY, {}, config.stage(Stage.SUMMARY))
    assert result.status == ExecutionStatus.NOT_INTEGRATED
    assert result.is_success is False

def test_classification_is_enabled_and_loadable():
    config = load_config()
    assert config.stage(Stage.CLASSIFICATION).enabled is True
    registry = build_agent_registry(config, {})
    adapter = registry[Stage.CLASSIFICATION]
    assert not getattr(adapter, "not_integrated", False)


def test_extraction_is_enabled_and_loadable():
    config = load_config()
    assert config.stage(Stage.EXTRACTION).enabled is True
    registry = build_agent_registry(config, {})
    adapter = registry[Stage.EXTRACTION]
    assert not getattr(adapter, "not_integrated", False)


def test_rag_is_enabled_and_loadable():
    config = load_config()
    assert config.stage(Stage.RAG).enabled is True
    registry = build_agent_registry(config, {})
    adapter = registry[Stage.RAG]
    assert not getattr(adapter, "not_integrated", False)
