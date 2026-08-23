"""Integrate real RoutingAgent with the Orchestration workflow state."""

from Agents.routing_agent.agent import RoutingAgent
from Orchestration.graph.graph_definition import Stage
from Orchestration.tests.mock_agents import (
    MockClassificationAgent,
    MockExtractionAgent,
    MockOCRAgent,
    MockRagAgent,
    MockSummaryAgent,
    MockValidationAgent,
    MockWriterAgent,
)
from Orchestration.workflow.workflow_builder import build_workflow


class _ITSupportOCR(MockOCRAgent):
    """OCR mock whose text triggers Bilgi İşlem routing."""

    def run(self, state):
        state = super().run(state)
        text = "Sunucularda sistem arızası var, teknik destek talep ediyorum."
        state["ocr_result"] = {
            "Success": True,
            "Data": [{"document_id": state.get("document_id"), "full_text": text}],
        }
        return state


class _ITSupportClassification(MockClassificationAgent):
    def run(self, state):
        state["classification_status"] = "completed"
        state["classification_result"] = {
            "document_type": "teknik destek",
            "doc_type": "teknik destek",
            "classification_confidence": 0.9,
        }
        return state


class _ITSupportSummary(MockSummaryAgent):
    def run(self, state):
        state["summary"] = {
            "success": True,
            "rag_summary_text": "Sistem arızası için teknik destek talebi.",
        }
        return state


def test_real_routing_is_integrated_with_orchestration_state():
    workflow = build_workflow(
        agent_overrides={
            Stage.OCR: _ITSupportOCR(),
            Stage.CLASSIFICATION: _ITSupportClassification(),
            Stage.EXTRACTION: MockExtractionAgent(),
            Stage.VALIDATION: MockValidationAgent(requires_rag=False),
            Stage.RAG: MockRagAgent(),
            Stage.SUMMARY: _ITSupportSummary(),
            Stage.ROUTING: RoutingAgent(),
            Stage.WRITING: MockWriterAgent(),
        }
    )

    result = workflow.run(
        {
            "document_id": "DOC-002",
            "question": "Sistemde arıza var, teknik destek istiyorum.",
        }
    )

    assert result.completed is True
    assert result.state["stage_status"]["routing"] == "success"
    assert result.state["routing"] == {
        "success": True,
        "department": "Bilgi İşlem Daire Başkanlığı",
    }


def test_routing_reads_graph_state_result_sections():
    """Direct contract: GraphState *_result / summary → routing.{success,department}."""
    agent = RoutingAgent()
    state = {
        "document_id": "DOC-002",
        "question": "Sistemde arıza var, teknik destek istiyorum.",
        "request": {
            "question": "Sistemde arıza var, teknik destek istiyorum.",
            "document_id": "DOC-002",
        },
        "ocr_result": {
            "Success": True,
            "Data": [
                {
                    "full_text": "Sunucularda sistem arızası var, teknik destek talep ediyorum.",
                }
            ],
        },
        "classification_result": {
            "document_type": "teknik destek",
            "classification_confidence": 0.9,
        },
        "summary": {
            "success": True,
            "rag_summary_text": "Sistem arızası için teknik destek talebi.",
        },
        "routing": {},
    }

    updated = agent.run(state)
    assert updated["routing"]["success"] is True
    assert updated["routing"]["department"] == "Bilgi İşlem Daire Başkanlığı"
    # Passthrough: upstream sections untouched.
    assert updated["summary"] == state["summary"]
    assert updated["ocr_result"] == state["ocr_result"]
