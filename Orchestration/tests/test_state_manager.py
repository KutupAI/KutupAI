from Orchestration.graph.graph_definition import Stage
from Orchestration.messages.message_schema import AgentResult, ExecutionStatus
from Orchestration.state.state_manager import StateManager


def test_initialize_sets_required_keys():
    mgr = StateManager(workflow_id="wf-1")
    state = mgr.initialize({"document_id": "doc-1", "document_path": "/tmp/a.pdf"})

    assert state["workflow_id"] == "wf-1"
    assert state["document_id"] == "doc-1"
    assert state["document_path"] == "/tmp/a.pdf"
    assert state["stage_status"] == {}
    assert state["stage_retries"] == {}
    assert state["history"] == []
    assert state["errors"] == []
    assert mgr.validate(state) == []


def test_initialize_generates_document_id_when_missing():
    mgr = StateManager()
    state = mgr.initialize({})
    assert state["document_id"]  # non-empty generated id


def test_apply_result_success_writes_only_own_section():
    mgr = StateManager()
    state = mgr.initialize({"document_id": "doc-1"})
    result = AgentResult.ok(Stage.CLASSIFICATION.value, {"doc_type": "invoice"})

    mgr.apply_result(state, Stage.CLASSIFICATION, result, attempt=1)

    assert state["classification_result"] == {"doc_type": "invoice"}
    assert "extraction_result" not in state
    assert state["stage_status"][Stage.CLASSIFICATION.value] == "success"
    assert len(state["history"]) == 1
    assert state["history"][0]["stage"] == "classification"
    assert state["errors"] == []


def test_apply_result_failure_records_error():
    mgr = StateManager()
    state = mgr.initialize({"document_id": "doc-1"})
    result = AgentResult.fail(Stage.VALIDATION.value, ExecutionStatus.FAILURE, "bad data")

    mgr.apply_result(state, Stage.VALIDATION, result, attempt=1)

    assert state["stage_status"][Stage.VALIDATION.value] == "failure"
    assert len(state["errors"]) == 1
    assert state["errors"][0]["stage"] == "validation"
    assert state["errors"][0]["error_type"] == "failure"


def test_terminate_and_finalize():
    mgr = StateManager()
    state = mgr.initialize({"document_id": "doc-1"})
    mgr.terminate(state, "unrecoverable failure")
    mgr.finalize(state, {"completed": False, "terminated": True})

    assert state["terminated"] is True
    assert state["termination_reason"] == "unrecoverable failure"
    assert state["final_decision"]["terminated"] is True
    assert state["current_agent"] is None


def test_snapshot_redacts_sensitive_fields():
    mgr = StateManager()
    state = mgr.initialize({"document_id": "doc-1", "accompanying_text": "secret contents"})
    snap = StateManager.snapshot(state)
    assert snap["accompanying_text"] == "<redacted>"
    assert snap["document_id"] == "doc-1"
