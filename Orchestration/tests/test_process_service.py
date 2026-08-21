import tempfile
from pathlib import Path

from Orchestration.graph.graph_definition import Stage
from Orchestration.process_service import run_full_workflow, run_ocr_pipeline

from Orchestration.tests.mock_agents import (
    MockClassificationAgent,
    MockExtractionAgent,
    MockOCRAgent,
    MockRagAgent,
    MockRoutingAgent,
    MockSummaryAgent,
    MockValidationAgent,
    MockWriterAgent,
)


def test_run_ocr_pipeline_missing_document_path_returns_failure_envelope():
    envelope = run_ocr_pipeline(document_id="doc-1", document_path=None)
    assert envelope["Success"] is False
    assert envelope["Data"][0]["document_id"] == "doc-1"


def test_run_ocr_pipeline_missing_file_returns_failure_envelope():
    envelope = run_ocr_pipeline(document_id="doc-1", document_path="/tmp/does-not-exist-xyz.pdf")
    assert envelope["Success"] is False


def test_run_ocr_pipeline_success_with_mock_agent():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 fake")
        path = f.name
    try:
        envelope = run_ocr_pipeline(document_id="doc-1", document_path=path, agent=MockOCRAgent())
        assert envelope["Success"] is True
        assert envelope["Data"][0]["document_id"] == "doc-1"
        assert envelope["Data"][0]["full_text"] == "hello world"
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_full_workflow_success_with_all_mocks():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 fake")
        path = f.name
    try:
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
        envelope = run_full_workflow(
            document_id="doc-1", document_path=path, agent_overrides=overrides
        )
        assert envelope["Success"] is True
        doc = envelope["Data"][0]
        assert doc["document_id"] == "doc-1"
        assert doc["writing"] == "Dear Sir, ..."
        assert doc["routing"] == {"department": "finance"}
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_full_workflow_defaults_to_ocr_only():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 fake")
        path = f.name
    try:
        envelope = run_full_workflow(
            document_id="doc-1",
            document_path=path,
            agent_overrides={Stage.OCR: MockOCRAgent()},
        )
        # OCR-only by default config: workflow still completes (later
        # stages skip, none are faked).
        assert envelope["Success"] is True
        doc = envelope["Data"][0]
        assert "writing" not in doc
        assert "classification" not in doc
    finally:
        Path(path).unlink(missing_ok=True)
