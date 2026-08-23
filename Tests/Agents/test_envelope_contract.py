"""
test_envelope_contract.py
==========================

Verifies the Routing Agent against the pipeline envelope contract:

    routing input  : {request, ocr, classification, extraction,
                       validation, rag, summary, routing: {}, writing}
    routing output : same envelope with
                       routing: {"success": bool, "department": string}

See Layers_contracts/Agents-contract/Routing.md for the canonical example.
"""

from __future__ import annotations

import unittest

from Agents.routing_agent.agent import RoutingAgent, process
from Agents.routing_agent.knowledge_base import default_knowledge_base


def make_envelope(**overrides) -> dict:
    base = {
        "request": {
            "success": True,
            "question": "bu ne sozlesmesi",
            "document": {
                "document_id": "DOC-001",
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
                "full_text": "...",
                "vision": {
                    "signature": {"detected": True, "handwritten": True},
                    "stamp": {"detected": False},
                },
            },
        },
        "classification": {
            "success": True,
            "document_type": "Elektrik sozlesmesi",
            "classification_confidence": 0.95,
        },
        "extraction": {
            "success": True,
            "sender": None,
            "date": None,
            "address": None,
            "phone": None,
            "email": None,
        },
        "validation": {
            "success": True,
            "is_complete": False,
            "errors": [],
            "warnings": [],
        },
        "rag": {
            "success": True,
            "rag_data": {"operation": "retrieve", "query": "...", "results": []},
        },
        "summary": {"success": True, "rag_summary_text": "..."},
        "routing": {},
        "writing": {},
    }
    base.update(overrides)
    return base


class EnvelopeContractTests(unittest.TestCase):
    def setUp(self):
        self.agent = RoutingAgent(knowledge_base=default_knowledge_base())

    def test_output_shape_matches_contract(self):
        envelope = make_envelope()
        out = self.agent.process(envelope)

        # Envelope shape / passthrough
        for key in ("request", "ocr", "classification", "extraction",
                    "validation", "rag", "summary", "writing"):
            self.assertEqual(out[key], envelope[key])

        # routing contract: {success: bool, department: string}
        self.assertIn("routing", out)
        self.assertIsInstance(out["routing"], dict)
        self.assertSetEqual(set(out["routing"].keys()), {"success", "department"})
        self.assertIsInstance(out["routing"]["success"], bool)
        self.assertIsInstance(out["routing"]["department"], str)

    def test_functional_process_entrypoint_matches_method(self):
        envelope = make_envelope()
        out_fn = process(envelope, agent=self.agent)
        out_method = self.agent.process(envelope)
        self.assertEqual(out_fn["routing"], out_method["routing"])

    def test_real_signal_routes_to_expected_department(self):
        envelope = make_envelope(
            request={
                "success": True,
                "question": "Sistemde arıza var, teknik destek istiyorum.",
                "document": {"document_id": "DOC-002", "file_name": "arg.pdf", "file_type": "pdf"},
            },
            ocr={
                "success": True,
                "ocr_data": {
                    "page_count": 1,
                    "language": "tr",
                    "pages": [],
                    "full_text": "Sunucularda sistem arızası var, teknik destek talep ediyorum.",
                    "vision": {"signature": {"detected": False, "handwritten": False}, "stamp": {"detected": False}},
                },
            },
            classification={"success": True, "document_type": "teknik destek", "classification_confidence": 0.9},
        )
        out = self.agent.process(envelope)
        self.assertTrue(out["routing"]["success"])
        self.assertEqual(out["routing"]["department"], "Bilgi İşlem Daire Başkanlığı")

    def test_empty_upstream_data_yields_graceful_failure_not_crash(self):
        envelope = make_envelope(
            request={"success": False, "question": "", "document": {}},
            ocr={},
            classification={},
            extraction={},
            validation={},
            rag={},
            summary={},
        )
        out = self.agent.process(envelope)
        self.assertFalse(out["routing"]["success"])
        self.assertEqual(out["routing"]["department"], "")

    def test_missing_routing_writing_keys_do_not_crash(self):
        envelope = make_envelope()
        del envelope["routing"]
        del envelope["writing"]
        out = self.agent.process(envelope)
        self.assertIn("routing", out)
        self.assertNotIn("writing", out)  # passthrough only copies existing keys


if __name__ == "__main__":
    unittest.main()
