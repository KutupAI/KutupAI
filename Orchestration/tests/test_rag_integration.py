"""Integrate real RAGAgent with the Orchestration workflow state."""

from __future__ import annotations

from unittest.mock import patch

from Agents.rag_agent import RAGAgent, RAG_CONTRACT_KEYS
from Orchestration.graph.graph_definition import Stage
from Orchestration.tests.mock_agents import (
    MockClassificationAgent,
    MockExtractionAgent,
    MockOCRAgent,
    MockRoutingAgent,
    MockSummaryAgent,
    MockValidationAgent,
    MockWriterAgent,
)
from Orchestration.workflow.workflow_builder import build_workflow
from RAG.retriever.query_router import QueryPlan


class _ContractOCR(MockOCRAgent):
    """OCR mock with unified envelope text for RAG query building."""

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


def _fake_retrieve(*_args, **_kwargs):
    return [
        {
            "id": "chunk-1",
            "text": "MADDE 1 - Deneme hukmu",
            "score": 0.9,
            "metadata": {
                "chunk_id": "chunk-1",
                "law_number": "6446",
                "law_name": "Elektrik Piyasasi Kanunu",
                "article_no": "1",
                "page_start": 2,
                "page_end": 2,
            },
        }
    ]


def test_real_rag_is_integrated_with_orchestration_state():
    with (
        patch("RAG.client.contract_adapter.retrieve", side_effect=_fake_retrieve),
        patch(
            "RAG.client.contract_adapter.choose_query_plan",
            return_value=QueryPlan("semantic_fast", "vector", False, True, False, "test"),
        ),
    ):
        workflow = build_workflow(
            agent_overrides={
                Stage.OCR: _ContractOCR(),
                Stage.CLASSIFICATION: _ContractClassification(),
                Stage.EXTRACTION: MockExtractionAgent(),
                Stage.VALIDATION: MockValidationAgent(),
                Stage.RAG: RAGAgent(),
                Stage.SUMMARY: MockSummaryAgent(),
                Stage.ROUTING: MockRoutingAgent(),
                Stage.WRITING: MockWriterAgent(),
            }
        )
        result = workflow.run(
            {
                "document_id": "DOC-RAG-001",
                "question": "bu ne sozlesmesi",
                "document_path": "/tmp/Elektrik sozlesmesi.pdf",
                "success": True,
                "document": {
                    "document_id": "DOC-RAG-001",
                    "file_name": "Elektrik sozlesmesi.pdf",
                    "file_type": "pdf",
                },
            }
        )

    assert result.completed is True
    assert result.state["stage_status"]["rag"] == "success"

    rag = result.state["rag"]
    assert set(rag.keys()) >= set(RAG_CONTRACT_KEYS)
    assert rag["success"] is True
    assert rag["rag_data"]["operation"] == "retrieve"
    assert "bu ne sozlesmesi" in rag["rag_data"]["query"]
    assert rag["rag_data"]["results"][0]["chunk_id"] == "chunk-1"
    assert "answer" not in rag["rag_data"]

    wire = result.state["rag_result"]
    assert wire["success"] is True
    assert wire["data"] == rag["rag_data"]
    assert result.state["rag_status"] == "completed"


def test_rag_reads_unified_envelope_from_graph_state():
    """Direct contract: GraphState request/ocr/classification → rag slot."""
    agent = RAGAgent()
    state = {
        "document_id": "DOC-RAG-002",
        "request": {
            "success": True,
            "question": "bu ne sozlesmesi",
            "document": {
                "document_id": "DOC-RAG-002",
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
                "full_text": "ELEKTRIK SATIS SOZLESMESI abonelik sartlari",
            },
        },
        "classification": {
            "success": True,
            "document_type": "Elektrik sozlesmesi",
            "classification_confidence": 0.95,
        },
        "extraction": {"success": True},
        "validation": {"success": True},
        "rag": {},
    }

    with (
        patch("RAG.client.contract_adapter.retrieve", side_effect=_fake_retrieve),
        patch(
            "RAG.client.contract_adapter.choose_query_plan",
            return_value=QueryPlan("semantic_fast", "vector", False, True, False, "test"),
        ),
    ):
        updated = agent.run(state)

    assert updated["rag"]["success"] is True
    assert updated["rag"]["rag_data"]["results"][0]["law_number"] == "6446"
    assert updated["rag_result"]["data"] == updated["rag"]["rag_data"]
    assert updated["ocr"] == state["ocr"]
    assert updated["classification"] == state["classification"]
    assert "answer" not in updated["rag"]["rag_data"]
