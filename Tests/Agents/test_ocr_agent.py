"""
OCR Agent unit tests — no Paddle model download.

Run:
  python Tests/Agents/test_ocr_agent.py
  pytest Tests/Agents/test_ocr_agent.py -q
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_document_validation():
    from Agents.ocr_agent.document import validate_document

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scan.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        doc = validate_document(path, max_file_size_mb=10)
        assert doc.extension == ".png"
        assert doc.is_pdf is False

        bad = Path(tmp) / "note.docx"
        bad.write_bytes(b"x")
        try:
            validate_document(bad, max_file_size_mb=10)
            raise AssertionError("expected ValueError for unsupported format")
        except ValueError:
            pass
    print("OK document_validation")


def test_turkish_correction():
    from Agents.ocr_agent.core.correction import TurkishOCRCorrector

    c = TurkishOCRCorrector()
    d = c.correct("  Madde  17 ,  2024 ")
    assert d.text == "Madde 17, 2024"
    assert d.applied is True
    assert c.correct("").text == ""
    print("OK turkish_correction")


def test_document_insights():
    from Agents.ocr_agent.core.insights import build_insights

    text = (
        "Tarih: 10.08.2026\n"
        "MADDE 1- Birinci madde metni.\n"
        "Devam satırı.\n"
        "MADDE 2- İkinci madde.\n"
        "İmza:\n"
        "Mehmet Kaya\n"
    )
    insights = build_insights(text)
    assert insights.has_signature is True
    assert insights.has_handwritten_signature is True
    assert "Mehmet Kaya" in insights.signature_names
    assert insights.primary_date == "10.08.2026"
    assert insights.has_articles is True
    assert len(insights.articles) == 2
    assert insights.articles[0].number == "1"
    assert "Birinci madde metni." in insights.articles[0].lines
    assert "\\n" not in insights.lines[0]
    print("OK document_insights")


def test_ocr_parser():
    from Agents.ocr_agent.core.ocr_parser import OCRResultParser

    parser = OCRResultParser(confidence_threshold=0.3)
    raw = {
        "rec_texts": ["Merhaba", "low"],
        "rec_scores": [0.95, 0.1],
        "dt_polys": [
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            [[0, 0], [1, 0], [1, 1], [0, 1]],
        ],
    }
    items = parser.parse(raw, page_index=0)
    assert len(items) == 1
    assert items[0].text == "Merhaba"
    print("OK ocr_parser")


def test_layout_and_tables():
    from Agents.ocr_agent.core.layout import LayoutAnalyzer
    from Agents.ocr_agent.core.tables import TableExtractor

    layout, visuals = LayoutAnalyzer(0.3).analyze(
        {
            "layout_det_res": {
                "boxes": [
                    {"label": "text", "score": 0.9, "coordinate": [0, 0, 50, 20]},
                    {"label": "seal", "score": 0.8, "coordinate": [10, 10, 40, 40]},
                ]
            }
        },
        page_index=0,
    )
    assert len(layout) == 2
    assert any(v.element_type == "seal" for v in visuals)

    tables = TableExtractor().extract(
        {
            "table_res_list": [
                {
                    "pred_html": "<table><tr><td>A</td><td>B</td></tr></table>",
                    "bbox": [0, 0, 100, 50],
                    "score": 0.7,
                }
            ]
        },
        page_index=0,
    )
    assert len(tables) == 1
    assert tables[0].cells[0].text == "A"
    print("OK layout_and_tables")


def test_preprocessor_resize():
    import cv2
    from Agents.ocr_agent.config import PreprocessingConfig
    from Agents.ocr_agent.preprocessing.image_preprocessor import ImagePreprocessor

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)
    out = ImagePreprocessor(
        PreprocessingConfig(
            enabled=True,
            grayscale=True,
            denoise=False,
            contrast=False,
            sharpen=False,
            deskew=False,
            perspective_correction=False,
            auto_crop_document=False,
            pale_boost=False,
            min_dimension=0,
            max_dimension=50,
        )
    ).process(img)
    assert max(out.shape[:2]) <= 50
    print("OK preprocessor_resize")


def test_preprocessor_far_pale_tilt():
    """Far page on dark desk + pale ink + mild tilt should still produce OCR-ready image."""
    import cv2
    from Agents.ocr_agent.config import PreprocessingConfig
    from Agents.ocr_agent.preprocessing.image_preprocessor import ImagePreprocessor

    canvas = np.full((900, 700, 3), 30, dtype=np.uint8)  # dark desk
    page = np.full((500, 360, 3), 210, dtype=np.uint8)  # pale paper
    cv2.putText(page, "Tarih: 10.08.2026", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 1)
    cv2.putText(page, "Imza: Mehmet", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 1)
    # Place page small/far in frame
    canvas[180:680, 170:530] = page
    # Mild rotation to simulate phone tilt
    m = cv2.getRotationMatrix2D((350, 450), 8, 1.0)
    tilted = cv2.warpAffine(canvas, m, (700, 900), borderValue=(30, 30, 30))

    out = ImagePreprocessor(PreprocessingConfig()).process(tilted)
    assert out is not None and out.size > 0
    # Should upscale distant crop toward min_dimension
    assert min(out.shape[:2]) >= 1600
    # Pale boost should increase contrast vs original pale page
    assert float(out.std()) > float(page.std())
    print("OK preprocessor_far_pale_tilt")


def test_processor_with_mocked_engine():
    from Agents.ocr_agent.config import OCRConfig, PreprocessingConfig
    from Agents.ocr_agent.processing.processor import OCRProcessor

    engine = MagicMock()
    engine.predict.return_value = [
        {
            "rec_texts": ["Madde 1- Test"],
            "rec_scores": [0.99],
            "dt_polys": [[[0, 0], [20, 0], [20, 10], [0, 10]]],
            "layout_det_res": {"boxes": []},
            "table_res_list": [],
        }
    ]

    config = OCRConfig(
        preprocessing=PreprocessingConfig(
            enabled=False,
            perspective_correction=False,
            deskew=False,
        )
    )
    processor = OCRProcessor(config, engine=engine)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.png"
        import cv2

        cv2.imwrite(str(path), np.full((40, 80, 3), 255, dtype=np.uint8))
        result = processor.process(path, document_id="doc-1")

    assert result.success is True
    assert "Madde 1" in result.full_text
    assert result.document_id == "doc-1"
    assert result.has_articles is True
    assert result.lines
    payload = result.to_dict()
    assert payload["summary"]["has_articles"] is True
    assert isinstance(payload["lines"], list)
    engine.predict.assert_called()
    print("OK processor_mocked_engine")


def test_agent_run_success_and_missing_path():
    from Agents.base.agent_registry import clear_registry, get_agent, list_agents, register
    from Agents.ocr_agent.agent import OCRAgent
    from Agents.ocr_agent.models import UnifiedOCRResult

    clear_registry()
    register(OCRAgent)

    mock_client = MagicMock()
    mock_client.process.return_value = UnifiedOCRResult(
        success=True,
        document_id="d1",
        file_name="a.png",
        file_type="png",
        language="tr",
        pages=[],
        full_text="Merhaba Türkiye",
    )
    agent = OCRAgent(client=mock_client)
    out = agent.run({"document_path": "C:/tmp/a.png", "document_id": "d1"})
    assert out["ocr_status"] == "completed"
    assert out["document_text"] == "Merhaba Türkiye"
    assert out["ocr_result"]["success"] is True

    missing = OCRAgent(client=mock_client).run({})
    assert missing["ocr_status"] == "failed"
    assert missing["ocr_result"]["error"] == "missing_document_path"

    assert "ocr_agent" in list_agents()
    assert get_agent("ocr_agent").name == "ocr_agent"
    print("OK agent_run")


def test_pdf_renderer_page_limit():
    from Agents.ocr_agent.processing.pdf_renderer import PDFRenderer

    # Without a real PDF, only validate constructor wiring
    renderer = PDFRenderer(dpi=150, max_pages=2)
    assert renderer.dpi == 150
    assert renderer.max_pages == 2
    print("OK pdf_renderer_config")


if __name__ == "__main__":
    test_document_validation()
    test_turkish_correction()
    test_document_insights()
    test_ocr_parser()
    test_layout_and_tables()
    test_preprocessor_resize()
    test_preprocessor_far_pale_tilt()
    test_processor_with_mocked_engine()
    test_agent_run_success_and_missing_path()
    test_pdf_renderer_page_limit()
    print("All OCR agent tests passed.")
