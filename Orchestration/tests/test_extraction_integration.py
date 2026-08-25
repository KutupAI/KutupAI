"""Integrate real ExtractionAgent with the Orchestration workflow state."""

from __future__ import annotations

from Agents.extraction_agent.agent import EXTRACTION_CONTRACT_KEYS, ExtractionAgent
from Agents.extraction_agent.config import (
    ExtractionAgentConfig,
    LLMConfig,
    NERConfig,
    VisionConfig,
)
from Orchestration.graph.graph_definition import Stage
from Orchestration.tests.mock_agents import (
    MockClassificationAgent,
    MockOCRAgent,
    MockRagAgent,
    MockRoutingAgent,
    MockSummaryAgent,
    MockValidationAgent,
    MockWriterAgent,
)
from Orchestration.workflow.workflow_builder import build_workflow


class _ContractOCR(MockOCRAgent):
    """OCR mock with structured text + unified envelope for extraction."""

    def run(self, state):
        state = super().run(state)
        text = (
            "ELEKTRIK SATIS SOZLESMESI\n"
            "Tarih: 12.05.2024\n"
            "Ahmet Yilmaz\n"
            "Tel: 0532 123 45 67\n"
            "E-posta: ahmet@example.com\n"
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


class _ContractClassification(MockClassificationAgent):
    def run(self, state):
        payload = {
            "success": True,
            "document_type": "Elektrik sozlesmesi",
            "classification_confidence": 0.95,
        }
        state["classification_status"] = "completed"
        state["classification"] = payload
        state["classification_result"] = payload
        return state


def _offline_extraction_agent() -> ExtractionAgent:
    """Real agent, LLM/NER/vision off — regex fields only (no Inference)."""
    return ExtractionAgent(
        ExtractionAgentConfig(
            ner=NERConfig(enabled=False),
            llm=LLMConfig(enabled=False, use_langextract=False),
            vision=VisionConfig(enabled=False),
        )
    )


def test_real_extraction_is_integrated_with_orchestration_state():
    workflow = build_workflow(
        agent_overrides={
            Stage.OCR: _ContractOCR(),
            Stage.CLASSIFICATION: _ContractClassification(),
            Stage.EXTRACTION: _offline_extraction_agent(),
            Stage.VALIDATION: MockValidationAgent(),
            Stage.RAG: MockRagAgent(),
            Stage.SUMMARY: MockSummaryAgent(),
            Stage.ROUTING: MockRoutingAgent(),
            Stage.WRITING: MockWriterAgent(),
        }
    )
    result = workflow.run(
        {
            "document_id": "DOC-EXT-001",
            "question": "bu ne sozlesmesi",
            "document_path": "/tmp/Elektrik sozlesmesi.pdf",
        }
    )

    assert result.completed is True
    assert result.state["stage_status"]["extraction"] == "success"

    extraction = result.state["extraction"]
    assert set(extraction.keys()) == set(EXTRACTION_CONTRACT_KEYS)
    assert extraction["success"] is True
    assert extraction["date"] == "12.05.2024"
    assert extraction["phone"] == "0532 123 45 67"
    assert extraction["email"] == "ahmet@example.com"
    assert extraction["sender"] is None
    assert extraction["address"] is None
    assert result.state["extraction_result"] == extraction
    assert result.state["extraction_status"] == "completed"


def test_extraction_reads_unified_envelope_from_graph_state():
    """Direct contract: GraphState ocr + classification → extraction contract."""
    agent = _offline_extraction_agent()
    state = {
        "document_id": "DOC-EXT-002",
        "request": {
            "success": True,
            "question": "bu ne sozlesmesi",
            "document": {
                "document_id": "DOC-EXT-002",
                "file_name": "Elektrik sozlesmesi.pdf",
                "file_type": "pdf",
            },
        },
        "ocr": {
            "success": True,
            "ocr_data": {
                "page_count": 1,
                "language": "tr",
                "pages": [],
                "full_text": "Tarih: 01.01.2025\nmail: test@ornek.com",
                "vision": {
                    "signature": {"detected": False, "handwritten": False},
                    "stamp": {"detected": False},
                },
            },
        },
        "classification": {
            "success": True,
            "document_type": "Elektrik sozlesmesi",
            "classification_confidence": 0.95,
        },
        "extraction": {},
    }

    updated = agent.run(state)

    assert set(updated["extraction"].keys()) == set(EXTRACTION_CONTRACT_KEYS)
    assert updated["extraction"]["success"] is True
    assert updated["extraction"]["date"] == "01.01.2025"
    assert updated["extraction"]["email"] == "test@ornek.com"
    assert updated["extraction_result"] == updated["extraction"]
    assert updated["ocr"] == state["ocr"]
    assert updated["classification"] == state["classification"]
