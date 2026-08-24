"""Integrate real ValidationAgent with the Orchestration workflow state."""

from Agents.validation_agent.agent import ValidationAgent
from Orchestration.graph.graph_definition import Stage
from Orchestration.tests.mock_agents import (
    MockClassificationAgent,
    MockOCRAgent,
    MockRagAgent,
    MockRoutingAgent,
    MockSummaryAgent,
    MockWriterAgent,
)
from Orchestration.workflow.workflow_builder import build_workflow


class _ContractOCR(MockOCRAgent):
    """OCR mock whose wire envelope ValidationAgent can adapt."""

    def run(self, state):
        state = super().run(state)
        state["ocr_result"] = {
            "Success": True,
            "Data": [
                {
                    "document_id": state.get("document_id"),
                    "full_text": "Ahmet Yilmaz, 01.03.2026, Istanbul, 5551234567, ahmet@example.com",
                    "page_count": 1,
                    "pages": [],
                }
            ],
        }
        return state


class _ContractClassification(MockClassificationAgent):
    def run(self, state):
        state["classification_status"] = "completed"
        state["classification_result"] = {
            "success": True,
            "document_type": "dilekce",
            "confidence": 0.92,
            "status": "success",
        }
        return state


class _ContractExtraction:
    """Writes nested extraction_result (what StateManager keeps after Extraction)."""

    def run(self, state):
        state["extraction_status"] = "completed"
        state["extraction_result"] = {
            "document": {"tarih": {"value": "01.03.2026", "confidence": 0.9, "source": "regex"}},
            "entities": {
                "person": {
                    "ad_soyad": [{"value": "Ahmet Yilmaz", "confidence": 0.9, "source": "ner"}],
                    "telefon": [{"value": "5551234567", "confidence": 0.9, "source": "regex"}],
                    "eposta": [{"value": "ahmet@example.com", "confidence": 0.9, "source": "regex"}],
                    "adres": [{"value": "Istanbul", "confidence": 0.8, "source": "ner"}],
                },
                "organization": {"kurum": [], "mudurluk": [], "ilgili_birim": []},
            },
            "request": {},
            "vision": {},
            "meta": {"success": True, "overall_confidence": 0.9},
        }
        return state


def test_real_validation_is_integrated_with_orchestration_state():
    workflow = build_workflow(
        agent_overrides={
            Stage.OCR: _ContractOCR(),
            Stage.CLASSIFICATION: _ContractClassification(),
            Stage.EXTRACTION: _ContractExtraction(),
            Stage.VALIDATION: ValidationAgent(),
            Stage.RAG: MockRagAgent(),
            Stage.SUMMARY: MockSummaryAgent(),
            Stage.ROUTING: MockRoutingAgent(),
            Stage.WRITING: MockWriterAgent(),
        }
    )

    result = workflow.run(
        {
            "document_id": "DOC-VAL-001",
            "question": "bu dilekce nedir",
        }
    )

    assert result.completed is True
    assert result.state["stage_status"]["validation"] == "success"

    validation = result.state["validation_result"]
    assert set(validation.keys()) == {"success", "is_complete", "errors", "warnings"}
    assert validation["success"] is True
    assert validation["is_complete"] is True
    assert validation["errors"] == []
    assert validation["warnings"] == []


def test_validation_reads_graph_state_result_sections():
    """Direct contract: GraphState *_result → validation.{success,is_complete,...}."""
    agent = ValidationAgent()
    state = {
        "document_id": "DOC-VAL-002",
        "ocr_result": {
            "Success": True,
            "Data": [{"full_text": "Merhaba", "page_count": 1, "pages": []}],
        },
        "classification_result": {
            "success": True,
            "document_type": "dilekce",
            "confidence": 0.95,
            "status": "success",
        },
        "extraction_result": {
            "meta": {"success": True},
            "document": {"tarih": {"value": None}},
            "entities": {
                "person": {
                    "ad_soyad": [],
                    "telefon": [],
                    "eposta": [],
                    "adres": [],
                }
            },
        },
        "validation": {},
    }

    updated = agent.run(state)
    assert updated["validation"]["success"] is True
    assert updated["validation"]["is_complete"] is False
    assert updated["validation"]["errors"] == []
    assert updated["validation_result"] == updated["validation"]
    # Passthrough: upstream wire sections untouched.
    assert updated["ocr_result"] == state["ocr_result"]
    assert updated["classification_result"] == state["classification_result"]
    assert updated["extraction_result"] == state["extraction_result"]


def test_validation_flags_extraction_failure_from_wire_keys():
    agent = ValidationAgent()
    state = {
        "ocr_result": {"Success": True, "Data": [{"full_text": "x", "pages": []}]},
        "classification_result": {"success": True, "confidence": 0.9, "document_type": "dilekce"},
        "extraction_result": {"meta": {"success": False}, "document": {}, "entities": {"person": {}}},
    }
    updated = agent.run(state)
    assert "extraction_failed" in updated["validation"]["errors"]
    assert updated["validation"]["success"] is False
    assert updated["validation_result"]["success"] is False
