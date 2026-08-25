"""Integrate real ClassificationAgent with the Orchestration workflow state."""

from __future__ import annotations

from unittest.mock import patch

from Agents.classification_agent.agent import ClassificationAgent
from Agents.classification_agent.config import ClassificationConfig
from Orchestration.graph.graph_definition import Stage
from Orchestration.tests.mock_agents import (
    MockExtractionAgent,
    MockOCRAgent,
    MockRagAgent,
    MockRoutingAgent,
    MockSummaryAgent,
    MockValidationAgent,
    MockWriterAgent,
)
from Orchestration.workflow.workflow_builder import build_workflow


class _ContractOCR(MockOCRAgent):
    """OCR mock with enough text for classification + unified-compatible wire."""

    def run(self, state):
        state = super().run(state)
        text = (
            "ELEKTRIK SATIS SOZLESMESI\n"
            "Taraflar arasinda elektrik enerjisi tedariki icin sozlesme."
        )
        state["ocr_result"] = {
            "Success": True,
            "Data": [
                {
                    "document_id": state.get("document_id"),
                    "file_name": "Elektrik sozlesmesi.pdf",
                    "file_type": "pdf",
                    "full_text": text,
                    "page_count": 1,
                    "pages": [],
                }
            ],
        }
        state["ocr"] = {
            "success": True,
            "ocr_data": {
                "page_count": 1,
                "language": "tr",
                "pages": [],
                "full_text": text,
                "vision": {
                    "signature": {"detected": True, "handwritten": True},
                    "stamp": {"detected": False},
                },
            },
        }
        state["document_text"] = text
        return state


def test_real_classification_is_integrated_with_orchestration_state():
    agent = ClassificationAgent(
        ClassificationConfig(use_fast_classifier=False, needs_review_threshold=0.60)
    )

    with patch(
        "Agents.classification_agent.agent.run_vlm_classification",
        return_value={
            "document_type": "Elektrik sozlesmesi",
            "confidence": 0.95,
            "alternatives": [],
        },
    ):
        workflow = build_workflow(
            agent_overrides={
                Stage.OCR: _ContractOCR(),
                Stage.CLASSIFICATION: agent,
                Stage.EXTRACTION: MockExtractionAgent(),
                Stage.VALIDATION: MockValidationAgent(),
                Stage.RAG: MockRagAgent(),
                Stage.SUMMARY: MockSummaryAgent(),
                Stage.ROUTING: MockRoutingAgent(),
                Stage.WRITING: MockWriterAgent(),
            }
        )
        result = workflow.run(
            {
                "document_id": "DOC-CLS-001",
                "question": "bu ne sozlesmesi",
                "document_path": "/tmp/Elektrik sozlesmesi.pdf",
            }
        )

    assert result.completed is True
    assert result.state["stage_status"]["classification"] == "success"

    classification = result.state["classification"]
    assert set(classification.keys()) == {
        "success",
        "document_type",
        "classification_confidence",
    }
    assert classification == {
        "success": True,
        "document_type": "Elektrik sozlesmesi",
        "classification_confidence": 0.95,
    }
    assert result.state["classification_result"] == classification


def test_classification_reads_ocr_wire_from_graph_state():
    """Direct contract: GraphState ocr_result → classification contract."""
    agent = ClassificationAgent(
        ClassificationConfig(use_fast_classifier=False, needs_review_threshold=0.60)
    )
    state = {
        "document_id": "DOC-CLS-002",
        "request": {
            "success": True,
            "question": "bu ne sozlesmesi",
            "document": {
                "document_id": "DOC-CLS-002",
                "file_name": "Elektrik sozlesmesi.pdf",
                "file_type": "pdf",
            },
        },
        "ocr_result": {
            "Success": True,
            "Data": [
                {
                    "document_id": "DOC-CLS-002",
                    "full_text": "ELEKTRIK SATIS SOZLESMESI metni",
                    "pages": [],
                }
            ],
        },
        "classification": {},
    }

    with patch(
        "Agents.classification_agent.agent.run_vlm_classification",
        return_value={
            "document_type": "Elektrik sozlesmesi",
            "confidence": 0.91,
            "alternatives": [],
        },
    ):
        updated = agent.run(state)

    assert updated["classification"]["success"] is True
    assert updated["classification"]["document_type"] == "Elektrik sozlesmesi"
    assert updated["classification"]["classification_confidence"] == 0.91
    assert updated["classification_result"] == updated["classification"]
    assert updated["ocr_result"] == state["ocr_result"]
