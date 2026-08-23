"""
Independent test for the Writer Agent.

Runs the real WriterAgent implementation (Agents/writer_agent/agent.py)
against a valid Unified State. The only thing mocked is the network
call inside LlamaClient.generate -- Writer Agent's own code (context
extraction, prompt building, state handling) runs unmodified, exactly
as Orchestration would call it.
"""

import unittest
from unittest.mock import patch

from Agents.writer_agent.agent import WriterAgent
from Inference.client.inference_response import InferenceResponse


def make_state(**overrides) -> dict:
    state = {
        "request": {
            "success": True,
            "question": "What department handles this application?",
            "document": {
                "document_id": "doc-1",
                "file_name": "application.pdf",
                "file_type": "pdf",
            },
        },
        "ocr": {"success": True, "ocr_data": {}},
        "classification": {
            "success": True,
            "document_type": "residence_permit_application",
            "classification_confidence": 0.95,
        },
        "extraction": {
            "success": True,
            "sender": "Jane Doe",
            "date": "2026-01-15",
            "address": None,
            "phone": None,
            "email": None,
        },
        "validation": {
            "success": True,
            "is_complete": False,
            "errors": [],
            "warnings": ["Missing phone number"],
        },
        "rag": {"success": True, "rag_data": {}},
        "summary": {
            "success": True,
            "rag_summary_text": (
                "Residence permit applications are routed to the "
                "Foreigners' Department."
            ),
        },
        "routing": {"success": True, "department": ""},
        "writing": {},
    }
    state.update(overrides)
    return state


class WriterAgentTests(unittest.TestCase):
    @patch("Agents.writer_agent.agent.LlamaClient.generate")
    def test_valid_state_produces_answer(self, mock_generate):
        mock_generate.return_value = InferenceResponse(
            success=True,
            text="Your application should be routed to the Foreigners' Department.",
            model="gemma3",
            prompt_tokens=42,
            completion_tokens=12,
            total_tokens=54,
            finish_reason="stop",
        )

        agent = WriterAgent()
        state = make_state()
        result = agent.run(state)

        # Inference was called through the existing interface.
        self.assertTrue(mock_generate.called)

        # Output contract.
        self.assertIn("writing", result)
        self.assertTrue(result["writing"]["success"])
        self.assertEqual(
            result["writing"]["answer"],
            "Your application should be routed to the Foreigners' Department.",
        )

        # Rest of the Unified State is preserved untouched.
        self.assertEqual(result["classification"]["document_type"],
                          "residence_permit_application")
        self.assertEqual(result["request"]["question"],
                          "What department handles this application?")

    @patch("Agents.writer_agent.agent.LlamaClient.generate")
    def test_prompt_uses_summary_and_question(self, mock_generate):
        mock_generate.return_value = InferenceResponse(success=True, text="ok")

        agent = WriterAgent()
        agent.run(make_state())

        sent_request = mock_generate.call_args[0][0]
        system_msg, user_msg = sent_request.messages

        self.assertEqual(system_msg.role, "system")
        self.assertIn("Writer Agent", system_msg.content)

        self.assertIn("What department handles this application?", user_msg.content)
        self.assertIn("Foreigners' Department", user_msg.content)
        self.assertIn("residence_permit_application", user_msg.content)

    @patch("Agents.writer_agent.agent.LlamaClient.generate")
    def test_inference_failure_returns_failed_writing(self, mock_generate):
        mock_generate.return_value = InferenceResponse(
            success=False, text="", error="connection refused"
        )

        agent = WriterAgent()
        result = agent.run(make_state())

        self.assertEqual(result["writing"], {"success": False, "answer": ""})

    @patch("Agents.writer_agent.agent.LlamaClient.generate")
    def test_missing_question_short_circuits_without_calling_inference(
        self, mock_generate
    ):
        state = make_state(request={"success": True, "question": "", "document": {}})

        agent = WriterAgent()
        result = agent.run(state)

        mock_generate.assert_not_called()
        self.assertEqual(result["writing"], {"success": False, "answer": ""})

    def test_non_dict_state_raises_type_error(self):
        agent = WriterAgent()
        with self.assertRaises(TypeError):
            agent.run("not a state")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
