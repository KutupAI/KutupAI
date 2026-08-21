import importlib


MODULES = [
    "Orchestration",
    "Orchestration.graph.graph_definition",
    "Orchestration.messages.message_schema",
    "Orchestration.state.graph_state",
    "Orchestration.state.state_manager",
    "Orchestration.supervisor.routing_logic",
    "Orchestration.supervisor.supervisor_agent",
    "Orchestration.supervisor.supervisor_prompts",
    "Orchestration.workflow.workflow_config",
    "Orchestration.workflow.workflow_builder",
    "Orchestration.process_service",
]


def test_every_orchestration_module_imports_cleanly():
    for name in MODULES:
        importlib.import_module(name)


def test_process_service_does_not_require_agents_package_at_import_time():
    # process_service.py must stay importable even when Agents/ isn't
    # installed (it only imports Agents.ocr_agent lazily, inside
    # run_ocr_pipeline, when no `agent` override is supplied).
    module = importlib.import_module("Orchestration.process_service")
    assert hasattr(module, "run_ocr_pipeline")
    assert hasattr(module, "run_full_workflow")


def test_main_module_imports_cleanly():
    # main.py requires fastapi/uvicorn/pydantic to be installed, same as
    # any FastAPI entry point; it does not require Agents/ at import time.
    importlib.import_module("Orchestration.main")
