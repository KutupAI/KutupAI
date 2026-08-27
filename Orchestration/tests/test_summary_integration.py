"""Integrate real SummaryAgent with the Orchestration workflow state."""

from Agents.summary_agent.agent import SummaryAgent
from Inference.client.inference_response import InferenceResponse
from Orchestration.graph.graph_definition import Stage
from Orchestration.tests.mock_agents import (
    MockClassificationAgent,
    MockExtractionAgent,
    MockOCRAgent,
    MockRagAgent,
    MockRoutingAgent,
    MockValidationAgent,
    MockWriterAgent,
)
from Orchestration.workflow.workflow_builder import build_workflow


class _Client:
    def __init__(self) -> None:
        self.request = None

    def generate(self, request):
        self.request = request
        return InferenceResponse(
            success=True,
            text="- İkamet izni başvurusu valiliğe yapılır. [6458/31]",
        )


def test_real_summary_is_integrated_with_orchestration_state():
    from Agents.summary_agent.client import SummaryClient

    client = _Client()
    agent = SummaryAgent(client=SummaryClient(llama_client=client))

    workflow = build_workflow(
        agent_overrides={
            Stage.OCR: MockOCRAgent(),
            Stage.CLASSIFICATION: MockClassificationAgent(),
            Stage.EXTRACTION: MockExtractionAgent(),
            Stage.VALIDATION: MockValidationAgent(),
            Stage.RAG: MockRagAgent(),
            Stage.SUMMARY: agent,
            Stage.ROUTING: MockRoutingAgent(),
            Stage.WRITING: MockWriterAgent(),
        }
    )

    result = workflow.run(
        {"document_id": "DOC-001", "question": "Yabancı kişi ikamet iznini nereden alır?"}
    )

    assert result.completed is True
    assert result.state["stage_status"]["summary"] == "success"
    assert result.state["summary"] == {
        "success": True,
        "rag_summary_text": "- İkamet izni başvurusu valiliğe yapılır. [6458/31]",
    }
    assert client.request is not None
    prompt = client.request.messages[0].content
    assert "context A" in prompt
    assert "6458" in prompt


def test_summary_reads_rag_result_section_from_graph_state():
    """Direct contract check: GraphState rag_result → summary.rag_summary_text."""
    from Agents.summary_agent.client import SummaryClient

    client = _Client()
    agent = SummaryAgent(client=SummaryClient(llama_client=client))

    state = {
        "request": {"question": "test question"},
        "rag_result": {
            "success": True,
            "data": {
                "operation": "retrieve",
                "query": "test question",
                "results": [
                    {
                        "chunk_id": "c1",
                        "text": "Madde 31 hükmü.",
                        "law_number": "6458",
                        "article_no": "31",
                    }
                ],
            },
        },
        "summary": {},
    }

    updated = agent.run(state)
    assert updated["summary"]["success"] is True
    assert "rag_summary_text" in updated["summary"]
    assert "Madde 31" in client.request.messages[0].content


def test_empty_rag_results_skip_summary_without_failing_workflow():
    agent = SummaryAgent()
    updated = agent.run(
        {
            "request": {"question": "test question"},
            "rag": {
                "success": True,
                "rag_data": {"operation": "retrieve", "query": "test question", "results": []},
            },
            "summary": {},
        }
    )

    assert updated["summary"] == {"success": True, "rag_summary_text": ""}
