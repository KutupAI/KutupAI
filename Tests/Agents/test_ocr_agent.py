"""Tests for Agents/ocr_agent.

Run from repo root:
    python Tests/Agents/test_ocr_agent.py
    # or
    pytest Tests/Agents/test_ocr_agent.py -q

No GPU, no model download: the PaddleOCR/PP-StructureV3 engine is
replaced with a small fake that mimics its output shape.
"""

from __future__ import annotations

from dataclasses import replace
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Agents.base.agent_registry import get_agent, list_agents  # noqa: E402
from Agents.ocr_agent import OCRAgent, OCRConfig  # noqa: E402
from Agents.ocr_agent.config import VisionFallbackConfig  # noqa: E402
from Agents.ocr_agent.processing.processor import OCRProcessor  # noqa: E402


class FakeEngine:
    """Stands in for PaddleStructureEngine.predict(); no model weights needed."""

    engine_name = "FakeEngine (test)"

    def __init__(self, texts=None, scores=None, layout=None):
        self.texts = texts if texts is not None else ["Örnek metin"]
        self.scores = scores if scores is not None else [0.92]
        self.layout = layout
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        n = len(self.texts)
        polys = [[[0, 0], [50, 0], [50, 20], [0, 20]] for _ in range(n)]
        raw = {"dt_polys": polys, "rec_texts": self.texts, "rec_scores": self.scores}
        if self.layout is not None:
            raw["layout_det_res"] = {"boxes": self.layout}
        return [raw]


class AlwaysBlankEngine:
    engine_name = "AlwaysBlankEngine"

    def predict(self, image):
        return [{"dt_polys": [], "rec_texts": [], "rec_scores": []}]


def _white_image(w=300, h=200):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _write_png(tmp_dir, name="page.png"):
    path = os.path.join(tmp_dir, name)
    cv2.imwrite(path, _white_image())
    return path


class RegistryTests(unittest.TestCase):
    def test_ocr_agent_is_registered(self):
        self.assertIn("ocr_agent", list_agents())
        self.assertIs(get_agent("ocr_agent"), OCRAgent)


class AgentContractTests(unittest.TestCase):
    def _config(self):
        return replace(
            OCRConfig.from_env(),
            vision_fallback=VisionFallbackConfig(enabled=False),
        )

    def test_missing_document_path_fails_gracefully(self):
        agent = OCRAgent()
        out = agent.run({"document_id": "doc-x"})

        self.assertEqual(out["ocr_status"], "failed")
        self.assertFalse(out["ocr"]["success"])
        self.assertEqual(out["ocr"]["error"]["code"], "FILE_CORRUPTED")
        # Orchestration wire: { Success, Data } — no nested error key.
        self.assertFalse(out["ocr_result"]["Success"])
        self.assertIsInstance(out["ocr_result"]["Data"], list)
        self.assertIn("errors", out)

    def test_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "file.exe")
            with open(bad, "wb") as f:
                f.write(b"not a document")
            processor = OCRProcessor(self._config(), engine=FakeEngine())
            out = processor.process(bad, "doc-bad")
            self.assertEqual(out["status"], "failed")
            self.assertEqual(out["error"]["code"], "UNSUPPORTED_FILE_TYPE")

    def test_empty_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            empty = os.path.join(d, "empty.png")
            open(empty, "wb").close()
            processor = OCRProcessor(self._config(), engine=FakeEngine())
            out = processor.process(empty, "doc-empty")
            self.assertEqual(out["status"], "failed")
            self.assertEqual(out["error"]["code"], "FILE_CORRUPTED")

    def test_happy_path_image_contract_shape(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_png(d)
            processor = OCRProcessor(
                self._config(),
                engine=FakeEngine(
                    texts=["T.C. İstanbul Valiliği"],
                    scores=[0.95],
                    layout=[{"label": "title", "score": 0.9, "coordinate": [0, 0, 50, 20]}],
                ),
            )
            out = processor.process(path, "doc-good")

            self.assertTrue(out["success"])
            self.assertEqual(out["status"], "complete")
            data = out["data"]
            self.assertEqual(data["page_count"], 1)
            self.assertIn("İstanbul", data["full_text"])

            page = data["pages"][0]
            self.assertEqual(page["page_number"], 1)
            self.assertIn("İstanbul", page["text"])
            self.assertIn("vision", page)
            self.assertIn("signature", page["vision"])
            self.assertIn("stamp", page["vision"])
            # Public page contract stays lean (text + vision only).
            for forbidden_page_key in ("blocks", "tables", "quality", "warnings"):
                self.assertNotIn(forbidden_page_key, page)
            # Contract must NOT leak semantic/downstream fields.
            for forbidden in ("classification", "extracted_data", "summary", "answer"):
                self.assertNotIn(forbidden, data)

    def test_agent_wire_keys_on_success(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_png(d)
            from Agents.ocr_agent.client import OCRClient

            config = self._config()
            client = OCRClient(
                config,
                processor=OCRProcessor(
                    config,
                    engine=FakeEngine(texts=["Resmi yazı örneği"], scores=[0.93]),
                ),
            )
            agent = OCRAgent(client=client, config=config)

            out = agent.run({
                "document_id": "doc-wire",
                "document_path": path,
                "request": {
                    "success": True,
                    "question": "bu belge nedir?",
                    "document": {
                        "document_id": "doc-wire",
                        "file_name": os.path.basename(path),
                        "file_type": "png",
                    },
                },
                "ocr": {},
                "classification": {},
                "extraction": {},
            })

            self.assertEqual(out["ocr_status"], "completed")
            self.assertTrue(out["ocr"]["success"])
            self.assertIn("örneği", out["ocr"]["ocr_data"]["full_text"].casefold())
            self.assertTrue(out["ocr_result"]["Success"])
            self.assertEqual(len(out["ocr_result"]["Data"]), 1)
            self.assertIn("document_text", out)
            # Downstream sections must remain untouched.
            self.assertEqual(out["classification"], {})
            self.assertEqual(out["extraction"], {})

    def test_low_confidence_triggers_retry_then_recovers(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_png(d)
            engine = FakeEngine(texts=["belirsiz"], scores=[0.1])

            original_predict = engine.predict

            def predict(image):
                out = original_predict(image)
                if engine.calls >= 2:
                    out[0]["rec_scores"] = [0.9]
                return out

            engine.predict = predict
            processor = OCRProcessor(self._config(), engine=engine)
            out = processor.process(path, "doc-retry")
            self.assertGreaterEqual(engine.calls, 2)
            self.assertTrue(out["success"])
            self.assertIn("belirsiz", out["data"]["full_text"])

    def test_turkish_characters_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_png(d)
            text = "çğıİöşü ÇĞIÖŞÜ"
            processor = OCRProcessor(self._config(), engine=FakeEngine(texts=[text], scores=[0.9]))
            out = processor.process(path, "doc-tr")
            self.assertIn("ç", out["data"]["full_text"])
            self.assertIn("İ", out["data"]["full_text"])
            self.assertEqual(out["data"]["language"]["detected"], "tr")

    def test_blank_page_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_png(d)
            processor = OCRProcessor(
                replace(self._config(), vision_fallback=VisionFallbackConfig(enabled=False)),
                engine=AlwaysBlankEngine(),
            )
            out = processor.process(path, "doc-blank")
            self.assertEqual(out["status"], "failed")
            self.assertFalse(out["success"])


class OfficeFormatTests(unittest.TestCase):
    def test_docx_extraction_if_available(self):
        try:
            import docx  # noqa: F401
        except ImportError:
            self.skipTest("python-docx not installed")

        import docx as docx_lib

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sample.docx")
            document = docx_lib.Document()
            document.add_paragraph("Test İçerik ç ğ ı ö ş ü")
            document.save(path)

            processor = OCRProcessor(OCRConfig.from_env(), engine=FakeEngine())
            out = processor.process(path, "doc-docx")
            self.assertEqual(out["status"], "complete")
            self.assertIn("İçerik", out["data"]["full_text"])


if __name__ == "__main__":
    unittest.main()
