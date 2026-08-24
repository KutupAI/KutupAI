"""
test_classification_agent.py
============================

Verifies Classification Agent against the unified pipeline envelope:

    classification input :
        {request, ocr, classification: {}, extraction, validation,
         rag, summary, routing, writing}

    classification output :
        same envelope with
        classification: {
            "success": bool,
            "document_type": str,
            "classification_confidence": float
        }

Uses mocked fast-classifier / VLM so no model server is required.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from Agents.classification_agent.agent import (
    CLASSIFICATION_CONTRACT_KEYS,
    ClassificationAgent,
    process,
)
from Agents.classification_agent.config import ClassificationConfig


ENVELOPE_SECTIONS = (
    "request",
    "ocr",
    "classification",
    "extraction",
    "validation",
    "rag",
    "summary",
    "routing",
    "writing",
)


def make_envelope(**overrides) -> dict:
    """Canonical mock envelope matching the unified Agents contract."""
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
                "full_text": (
                    "ELEKTRIK SATIS SOZLESMESI\n"
                    "Taraflar arasinda elektrik enerjisi tedariki icin "
                    "duzenlenmis sozlesme metni."
                ),
                "vision": {
                    "signature": {"detected": True, "handwritten": True},
                    "stamp": {"detected": False},
                },
            },
        },
        "classification": {},
        "extraction": {},
        "validation": {},
        "rag": {},
        "summary": {},
        "routing": {},
        "writing": {},
    }
    base.update(overrides)
    return base


def make_agent() -> ClassificationAgent:
    # Disable fast path so tests control the VLM mock only.
    return ClassificationAgent(
        ClassificationConfig(use_fast_classifier=False, needs_review_threshold=0.60)
    )


class ClassificationEnvelopeContractTests(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    @patch("Agents.classification_agent.agent.run_vlm_classification")
    def test_output_shape_matches_contract(self, mock_vlm):
        mock_vlm.return_value = {
            "document_type": "Elektrik sozlesmesi",
            "confidence": 0.95,
            "alternatives": [],
        }
        envelope = make_envelope()
        out = self.agent.process(envelope)

        for key in ENVELOPE_SECTIONS:
            self.assertIn(key, out)

        # Passthrough: upstream sections unchanged
        for key in ("request", "ocr", "extraction", "validation", "rag", "summary", "routing", "writing"):
            self.assertEqual(out[key], envelope[key])

        classification = out["classification"]
        self.assertIsInstance(classification, dict)
        self.assertEqual(set(classification.keys()), set(CLASSIFICATION_CONTRACT_KEYS))
        self.assertTrue(classification["success"])
        self.assertEqual(classification["document_type"], "Elektrik sozlesmesi")
        self.assertEqual(classification["classification_confidence"], 0.95)

        # Wire mirrors for Orchestration GraphState
        self.assertIn("classification_result", out)
        self.assertEqual(out["classification_status"], "success")

    @patch("Agents.classification_agent.agent.run_vlm_classification")
    def test_reads_ocr_from_unified_envelope(self, mock_vlm):
        mock_vlm.return_value = {
            "document_type": "Elektrik sozlesmesi",
            "confidence": 0.95,
            "alternatives": [],
        }
        envelope = make_envelope()
        # No legacy ocr_result / document_text — only unified ocr.ocr_data
        self.assertNotIn("ocr_result", envelope)
        self.assertNotIn("document_text", envelope)

        out = self.agent.run(envelope)
        self.assertTrue(out["classification"]["success"])
        mock_vlm.assert_called_once()
        kwargs = mock_vlm.call_args.kwargs
        self.assertIn("ELEKTRIK", kwargs["normalized_text"])
        # Vision from empty pages promoted so prompts get signature/stamp
        pages = kwargs["ocr_pages"]
        self.assertTrue(pages)
        self.assertTrue(pages[0]["vision"]["signature"]["detected"])

    @patch("Agents.classification_agent.agent.run_vlm_classification")
    def test_module_process_entrypoint_matches_method(self, mock_vlm):
        mock_vlm.return_value = {
            "document_type": "Elektrik sozlesmesi",
            "confidence": 0.95,
            "alternatives": [],
        }
        envelope = make_envelope()
        out_fn = process(envelope, agent=self.agent)
        out_method = self.agent.process(envelope)
        self.assertEqual(out_fn["classification"], out_method["classification"])

    def test_empty_ocr_yields_graceful_failure_not_crash(self):
        envelope = make_envelope(
            request={"success": False, "question": "", "document": {}},
            ocr={},
            classification={},
        )
        out = self.agent.run(envelope)
        classification = out["classification"]
        self.assertEqual(set(classification.keys()), set(CLASSIFICATION_CONTRACT_KEYS))
        self.assertFalse(classification["success"])
        self.assertIsNone(classification["document_type"])
        self.assertEqual(classification["classification_confidence"], 0.0)
        self.assertEqual(out["classification_status"], "failed")

    @patch("Agents.classification_agent.agent.run_vlm_classification")
    def test_legacy_ocr_result_still_works(self, mock_vlm):
        mock_vlm.return_value = {
            "document_type": "dilekce",
            "confidence": 0.88,
            "alternatives": [],
        }
        state = {
            "document_id": "DOC-LEGACY",
            "ocr_result": {
                "full_text": "Sayin yetkili, dilekce metni burada.",
                "pages": [],
            },
            "classification": {},
        }
        out = self.agent.run(state)
        self.assertTrue(out["classification"]["success"])
        self.assertEqual(out["classification"]["document_type"], "dilekce")
        self.assertEqual(out["classification"]["classification_confidence"], 0.88)

    @patch("Agents.classification_agent.agent.run_fast_classifier")
    def test_fast_classifier_path_writes_contract(self, mock_fast):
        class _Fast:
            document_type = "Elektrik sozlesmesi"
            confidence = 0.97
            alternatives = []

        mock_fast.return_value = _Fast()
        agent = ClassificationAgent(
            ClassificationConfig(
                use_fast_classifier=True,
                fast_classifier_escalation_threshold=0.75,
            )
        )
        out = agent.run(make_envelope())
        self.assertEqual(
            out["classification"],
            {
                "success": True,
                "document_type": "Elektrik sozlesmesi",
                "classification_confidence": 0.97,
            },
        )


if __name__ == "__main__":
    unittest.main()
