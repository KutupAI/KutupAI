from Agents.writer_agent.agent import WriterAgent
from Inference.client.inference_response import InferenceResponse
from Orchestration.graph.graph_definition import Stage
from Orchestration.tests.mock_agents import (
    MockClassificationAgent,
    MockExtractionAgent,
    MockOCRAgent,
    MockRagAgent,
    MockRoutingAgent,
    MockSummaryAgent,
    MockValidationAgent,
)
from Orchestration.workflow.workflow_builder import build_workflow


class _Client:
    def __init__(self):
        self.request = None

    def generate(self, request):
        self.request = request
        return InferenceResponse(success=True, text="Bu bir elektrik sözleşmesidir.")


def test_real_writer_is_integrated_with_orchestration_state():
    client = _Client()
    workflow = build_workflow(
        agent_overrides={
            Stage.OCR: MockOCRAgent(),
            Stage.CLASSIFICATION: MockClassificationAgent(),
            Stage.EXTRACTION: MockExtractionAgent(),
            Stage.VALIDATION: MockValidationAgent(),
            Stage.RAG: MockRagAgent(),
            Stage.SUMMARY: MockSummaryAgent(),
            Stage.ROUTING: MockRoutingAgent(),
            Stage.WRITING: WriterAgent(client=client),
        }
    )

    result = workflow.run({"document_id": "DOC-001", "question": "bu ne sözleşmesi"})

    assert result.completed is True
    assert result.state["stage_status"]["writing"] == "success"
    assert result.state["writing"] == {
        "success": True,
        "answer": "Bu bir elektrik sözleşmesidir.",
    }
    assert "draft_letter" not in result.state
    prompt = client.request.messages[1].content
    assert "invoice" in prompt
    assert "summary text" in prompt
