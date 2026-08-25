import tempfile
from pathlib import Path

from Orchestration.graph.graph_definition import Stage
from Orchestration.process_service import run_workflow

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


def test_run_workflow_missing_document_path_returns_failure_envelope():
    envelope = run_workflow(document_id="doc-1", document_path=None)
    assert envelope["Success"] is False
    assert envelope["Data"][0]["document_id"] == "doc-1"


def test_run_workflow_missing_file_returns_failure_envelope():
    envelope = run_workflow(document_id="doc-1", document_path="/tmp/does-not-exist-xyz.pdf")
    assert envelope["Success"] is False


def test_run_workflow_ocr_success_with_mock_agent():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 fake")
        path = f.name
    try:
        envelope = run_workflow(
            document_id="doc-1",
            document_path=path,
            agent_overrides={Stage.OCR: MockOCRAgent()},
        )
        assert envelope["Success"] is True
        assert envelope["Data"][0]["document_id"] == "doc-1"
        assert envelope["Data"][0]["full_text"] == "hello world"
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_workflow_success_with_all_mocks():
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
        envelope = run_workflow(
            document_id="doc-1", document_path=path, agent_overrides=overrides
        )
        assert envelope["Success"] is True
        doc = envelope["Data"][0]
        assert doc["document_id"] == "doc-1"
        assert doc["writing"] == {"success": True, "answer": "Dear Sir, ..."}
        assert doc["routing"] == {"success": True, "department": "finance"}
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_workflow_defaults_to_ocr_through_validation():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 fake")
        path = f.name
    try:
        envelope = run_workflow(
            document_id="doc-1",
            document_path=path,
            agent_overrides={
                Stage.OCR: MockOCRAgent(),
                # Enabled stages mocked to avoid live Inference.
                Stage.CLASSIFICATION: MockClassificationAgent(),
                Stage.EXTRACTION: MockExtractionAgent(),
            },
        )
        assert envelope["Success"] is True
        doc = envelope["Data"][0]
        assert "writing" not in doc
        assert doc["classification"] == {
            "success": True,
            "document_type": "invoice",
            "classification_confidence": 0.95,
        }
        assert doc["extraction"] == {
            "success": True,
            "sender": None,
            "date": None,
            "address": None,
            "phone": None,
            "email": None,
        }
        assert "validation" in doc
        assert set(doc["validation"].keys()) == {"success", "is_complete", "errors", "warnings"}
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_workflow_from_application_envelope():
    from Orchestration.process_service import run_workflow_from_application

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 fake")
        path = f.name
    try:
        payload = {
            "document_id": "DOC-001",
            "document_path": path,
            "request": {
                "success": True,
                "question": "bu ne sozlesmesi",
                "document": {
                    "document_id": "DOC-001",
                    "file_name": "Elektrik sozlesmesi.pdf",
                    "file_type": "pdf",
                    "document_path": path,
                },
            },
            "ocr": {},
            "classification": {},
            "extraction": {},
            "validation": {},
            "rag": {},
            "summary": {},
            "routing": {},
            "writing": {},
        }
        envelope = run_workflow_from_application(
            payload,
            agent_overrides={Stage.OCR: MockOCRAgent()},
        )
        assert envelope["Success"] is True
        assert envelope["Data"][0]["document_id"] == "DOC-001"
        assert envelope["Data"][0]["question"] == "bu ne sozlesmesi"
    finally:
        Path(path).unlink(missing_ok=True)
