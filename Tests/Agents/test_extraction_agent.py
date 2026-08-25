"""
test_extraction_agent.py
========================

Verifies Extraction Agent against the unified pipeline envelope:

    extraction input :
        {request, ocr, classification, extraction: {}, validation,
         rag, summary, routing, writing}

    extraction output :
        same envelope with
        extraction: {
            "success": bool,
            "sender": str | null,
            "date": str | null,
            "address": str | null,
            "phone": str | null,
            "email": str | null
        }

Uses mocked LLM so no model server is required.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from Agents.extraction_agent.agent import (
    EXTRACTION_CONTRACT_KEYS,
    ExtractionAgent,
    process,
)
from Agents.extraction_agent.config import ExtractionAgentConfig, LLMConfig, NERConfig, VisionConfig


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
                    "Tarih: 12.05.2024\n"
                    "Ahmet Yilmaz\n"
                    "Tel: 0532 123 45 67\n"
                    "E-posta: ahmet@example.com\n"
                    "Taraflar arasinda elektrik enerjisi tedariki icin "
                    "duzenlenmis sozlesme metni."
                ),
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
        "extraction": {},
        "validation": {},
        "rag": {},
        "summary": {},
        "routing": {},
        "writing": {},
    }
    base.update(overrides)
    return base


def make_agent() -> ExtractionAgent:
    # Disable LLM/NER/vision network calls; regex still runs on OCR text.
    cfg = ExtractionAgentConfig(
        ner=NERConfig(enabled=False),
        llm=LLMConfig(enabled=False, use_langextract=False),
        vision=VisionConfig(enabled=False),
    )
    return ExtractionAgent(cfg)


class ExtractionEnvelopeContractTests(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_output_shape_matches_contract(self):
        envelope = make_envelope()
        out = self.agent.process(envelope)

        for key in ENVELOPE_SECTIONS:
            self.assertIn(key, out)

        # Passthrough: upstream / downstream sections unchanged
        for key in (
            "request",
            "ocr",
            "classification",
            "validation",
            "rag",
            "summary",
            "routing",
            "writing",
        ):
            self.assertEqual(out[key], envelope[key])

        extraction = out["extraction"]
        self.assertIsInstance(extraction, dict)
        self.assertEqual(set(extraction.keys()), set(EXTRACTION_CONTRACT_KEYS))
        self.assertTrue(extraction["success"])
        self.assertEqual(extraction["date"], "12.05.2024")
        self.assertEqual(extraction["phone"], "0532 123 45 67")
        self.assertEqual(extraction["email"], "ahmet@example.com")
        # sender comes from LLM/NER (disabled here) -> null
        self.assertIsNone(extraction["sender"])
        self.assertIsNone(extraction["address"])

        # Wire mirror for Orchestration GraphState
        self.assertEqual(out["extraction_result"], extraction)

    def test_reads_ocr_and_classification_from_unified_envelope(self):
        envelope = make_envelope()
        self.assertNotIn("ocr_result", envelope)
        self.assertNotIn("classification_result", envelope)

        out = self.agent.run(envelope)
        self.assertTrue(out["extraction"]["success"])
        self.assertEqual(out["extraction"]["date"], "12.05.2024")

    def test_module_process_entrypoint_matches_method(self):
        envelope = make_envelope()
        out_fn = process(envelope, agent=self.agent)
        out_method = self.agent.process(envelope)
        self.assertEqual(out_fn["extraction"], out_method["extraction"])

    def test_empty_ocr_yields_graceful_failure_not_crash(self):
        envelope = make_envelope(
            request={"success": False, "question": "", "document": {}},
            ocr={},
            classification={},
            extraction={},
        )
        out = self.agent.run(envelope)
        extraction = out["extraction"]
        self.assertEqual(set(extraction.keys()), set(EXTRACTION_CONTRACT_KEYS))
        self.assertFalse(extraction["success"])
        self.assertIsNone(extraction["sender"])
        self.assertIsNone(extraction["date"])
        self.assertIsNone(extraction["address"])
        self.assertIsNone(extraction["phone"])
        self.assertIsNone(extraction["email"])

    def test_llm_persons_map_to_sender(self):
        cfg = ExtractionAgentConfig(
            ner=NERConfig(enabled=False),
            llm=LLMConfig(enabled=True, use_langextract=False),
            vision=VisionConfig(enabled=False),
        )
        agent = ExtractionAgent(cfg)
        mock_llm = MagicMock()
        mock_llm.extract.return_value = {
            "data": {
                "persons": ["Ahmet Yilmaz"],
                "persons_spans": [None],
                "organizations": [],
                "organizations_spans": [],
                "confidence": 0.9,
            },
            "used": True,
            "retried": False,
            "retry_count": 0,
            "error": None,
            "langextract_used": False,
        }
        agent.llm = mock_llm

        out = agent.process(make_envelope())
        self.assertEqual(
            out["extraction"],
            {
                "success": True,
                "sender": "Ahmet Yilmaz",
                "date": "12.05.2024",
                "address": None,
                "phone": "0532 123 45 67",
                "email": "ahmet@example.com",
            },
        )
        mock_llm.extract.assert_called_once()
        _args, kwargs = mock_llm.extract.call_args
        self.assertEqual(kwargs.get("classification_hint"), "Elektrik sozlesmesi")

    def test_legacy_ocr_result_still_works(self):
        state = {
            "document_id": "DOC-LEGACY",
            "ocr_result": {
                "full_text": "Tarih: 01.01.2025\nmail: test@ornek.com",
                "pages": [],
            },
            "classification": {
                "success": True,
                "document_type": "dilekce",
                "classification_confidence": 0.88,
            },
            "extraction": {},
        }
        out = self.agent.run(state)
        self.assertTrue(out["extraction"]["success"])
        self.assertEqual(out["extraction"]["date"], "01.01.2025")
        self.assertEqual(out["extraction"]["email"], "test@ornek.com")
        self.assertEqual(set(out["extraction"].keys()), set(EXTRACTION_CONTRACT_KEYS))


if __name__ == "__main__":
    unittest.main()
