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
    assert "summary text" in prompt  # from MockSummaryAgent.rag_summary_text


def test_writer_uses_evidence_summary_when_model_rejects_available_context():
    class RejectingClient:
        def generate(self, request):
            return InferenceResponse(
                success=True,
                text="The relevant information does not include the specific details requested.",
            )

    agent = WriterAgent(client=RejectingClient())
    updated = agent.run(
        {
            "request": {"question": "7547 hangi maddeleri etkiledi?"},
            "summary": {
                "success": True,
                "rag_summary_text": "- Etkilenen maddeler: 3, 12, 13, Geçici Madde 4.",
            },
        }
    )

    assert updated["writing"]["answer"] == "- Etkilenen maddeler: 3, 12, 13, Geçici Madde 4."


def test_writer_calculates_explicit_calendar_difference_from_context():
    agent = WriterAgent()
    updated = agent.run(
        {
            "request": {
                "question": "Bu düzenleme 7196 sayılı Kanundan kaç yıl, ay ve gün sonra yürürlüğe girmiştir?"
            },
            "summary": {"success": True, "rag_summary_text": "7196 yürürlük tarihi: 24/12/2019."},
            "conversation_memory": "Önceki cevap: 7547 yürürlük tarihi: 16/05/2025.",
        }
    )

    assert "5 yıl 4 ay 22 gün" in updated["writing"]["answer"]
