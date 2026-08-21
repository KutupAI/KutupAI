from Orchestration.graph.graph_definition import Stage
from Orchestration.messages.message_schema import ExecutionStatus
from Orchestration.state.state_manager import StateManager
from Orchestration.supervisor.routing_logic import Action, decide_after_execution, decide_before_execution
from Orchestration.workflow.workflow_config import load_config


def _fresh_state():
    return StateManager().initialize({"document_id": "doc-1"})


def test_before_execution_skips_unavailable_optional_stage():
    config = load_config()
    state = _fresh_state()
    decision = decide_before_execution(Stage.RAG, state, config, available=False)
    assert decision.action == Action.SKIP
    assert decision.next_stage == Stage.SUMMARY


def test_before_execution_terminates_when_required_stage_unavailable():
    config = load_config()
    state = _fresh_state()
    # OCR's fallback policy is "terminate" and it is not optional.
    decision = decide_before_execution(Stage.OCR, state, config, available=False)
    assert decision.action == Action.TERMINATE


def test_before_execution_continues_when_available():
    config = load_config()
    state = _fresh_state()
    decision = decide_before_execution(Stage.OCR, state, config, available=True)
    assert decision.action == Action.CONTINUE
    assert decision.next_stage == Stage.OCR


def test_after_execution_advances_on_success():
    config = load_config()
    state = _fresh_state()
    decision = decide_after_execution(Stage.OCR, ExecutionStatus.SUCCESS, state, config)
    assert decision.action == Action.CONTINUE
    assert decision.next_stage == Stage.CLASSIFICATION


def test_after_execution_completes_on_last_stage_success():
    config = load_config()
    state = _fresh_state()
    decision = decide_after_execution(Stage.WRITING, ExecutionStatus.SUCCESS, state, config)
    assert decision.action == Action.COMPLETE


def test_after_execution_retries_before_exhausting():
    config = load_config()
    state = _fresh_state()
    # ocr retries=1 -> first failure should retry
    decision = decide_after_execution(Stage.OCR, ExecutionStatus.FAILURE, state, config)
    assert decision.action == Action.RETRY
    assert decision.next_stage == Stage.OCR


def test_after_execution_terminates_when_retries_exhausted_and_policy_terminate():
    config = load_config()
    state = _fresh_state()
    state["stage_retries"][Stage.OCR.value] = 1  # already used the one retry
    decision = decide_after_execution(Stage.OCR, ExecutionStatus.FAILURE, state, config)
    assert decision.action == Action.TERMINATE


def test_after_execution_skips_when_retries_exhausted_and_policy_skip():
    config = load_config()
    state = _fresh_state()
    state["stage_retries"][Stage.RAG.value] = 1  # rag retries=1, already used
    decision = decide_after_execution(Stage.RAG, ExecutionStatus.FAILURE, state, config)
    assert decision.action == Action.SKIP
    assert decision.next_stage == Stage.SUMMARY


def test_validation_skips_rag_when_not_required():
    config = load_config()
    state = _fresh_state()
    state["validation_result"] = {"valid": True, "requires_rag": False}
    decision = decide_after_execution(Stage.VALIDATION, ExecutionStatus.SUCCESS, state, config)
    assert decision.action == Action.CONTINUE
    assert decision.next_stage == Stage.SUMMARY
