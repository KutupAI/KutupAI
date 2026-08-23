"""
Integration-style upload → OCR flow (Application contract simulated in Python).

Covers:
- temp file persist + path handoff
- optional accompanying text
- primary OCR success → Qwen not called
- quality fail → Qwen called
- Qwen unavailable → usable Unstructured result still returned
- temp cleanup
- UnifiedOCRResult shape

Does not require a live Qwen server (mocks).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _png(path: Path) -> Path:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return path


def _good_unified(text: str = "Clear document text content here." * 3):
    from Agents.ocr_agent.models import contract_envelope, empty_document

    doc = empty_document(
        document_id="doc-1",
        file_name="doc.png",
        file_type="png",
        question="",
    )
    doc["full_text"] = text
    doc["pages"] = [
        {
            "page_number": 1,
            "text": text,
            "tables": [],
            "vision": {
                "signature": {"detected": False, "handwritten": False},
                "stamp": {"detected": False},
            },
        }
    ]
    return contract_envelope(True, [doc])


def test_temp_file_handoff_and_cleanup_without_text():
    from Orchestration.graph.graph_definition import Stage
    from Orchestration.process_service import run_workflow

    temp_root = Path(tempfile.mkdtemp(prefix="sgai_upload_"))
    request_id = "req-no-text"
    try:
        dest = temp_root / request_id
        dest.mkdir(parents=True)
        file_path = _png(dest / "upload.png")
        assert file_path.exists()

        agent = MagicMock()
        agent.run.side_effect = lambda state: {
            **state,
            "ocr_status": "completed",
            "document_text": "hello",
            "ocr_result": _good_unified("hello"),
        }

        out = run_workflow(
            document_id=request_id,
            document_path=str(file_path),
            accompanying_text=None,
            agent_overrides={Stage.OCR: agent},
        )
        assert out["Success"] is True
        assert agent.run.call_args[0][0]["document_path"] == str(file_path)

        shutil.rmtree(dest)
        assert not dest.exists()
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    print("OK temp_handoff_cleanup_no_text")


def test_temp_file_with_accompanying_text():
    from Orchestration.graph.graph_definition import Stage
    from Orchestration.process_service import run_workflow

    temp_root = Path(tempfile.mkdtemp(prefix="sgai_upload_"))
    request_id = "req-with-text"
    try:
        dest = temp_root / request_id
        dest.mkdir(parents=True)
        file_path = _png(dest / "scan.png")

        agent = MagicMock()
        agent.run.side_effect = lambda state: {
            **state,
            "ocr_status": "completed",
            "document_text": "x",
            "ocr_result": _good_unified("x"),
        }
        out = run_workflow(
            document_id=request_id,
            document_path=str(file_path),
            accompanying_text="ek açıklama",
            agent_overrides={Stage.OCR: agent},
        )
        assert out["Success"] is True
        state = agent.run.call_args[0][0]
        assert state.get("accompanying_text") == "ek açıklama" or state.get("question") == "ek açıklama"
        shutil.rmtree(dest)
        assert not file_path.exists()
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    print("OK temp_with_text")


def test_primary_ocr_success_skips_qwen():
    from Agents.ocr_agent.config import OCRConfig, QwenVLConfig, QualityConfig
    from Agents.ocr_agent.models import NormalizedOCRResult, PageContent
    from Agents.ocr_agent.processing.processor import OCRProcessor

    ocr = MagicMock()
    ocr.process.return_value = NormalizedOCRResult(
        document_id="d",
        file_name="c.png",
        file_type="png",
        language="tr",
        pages=[
            PageContent(
                page_number=1,
                text=("Bu belge tamamen net ve okunabilir. " * 8).strip(),
                confidence=0.99,
            )
        ],
        success=True,
        element_count=20,
        has_layout=True,
    )
    vision = MagicMock()
    vision.is_available.return_value = True

    with tempfile.TemporaryDirectory() as tmp:
        path = _png(Path(tmp) / "clear.png")
        result = OCRProcessor(
            OCRConfig(
                quality=QualityConfig(min_text_length=20, min_chars_per_page=5),
                qwen=QwenVLConfig(enabled=True),
            ),
            ocr_provider=ocr,
            vision_provider=vision,
        ).process(path, document_id="clear", accompanying_text="optional note")

    assert result.success is True
    assert result.qwen_used is False
    vision.recover_pages.assert_not_called()
    payload = result.to_dict()
    assert payload["classification"]["document_type"] is None
    json.dumps(payload)
    print("OK primary_skips_qwen")


def test_quality_fail_calls_qwen_with_text():
    from Agents.ocr_agent.config import OCRConfig, QwenVLConfig, QualityConfig
    from Agents.ocr_agent.models import NormalizedOCRResult, PageContent, VisionInfo
    from Agents.ocr_agent.processing.processor import OCRProcessor

    ocr = MagicMock()
    ocr.process.return_value = NormalizedOCRResult(
        document_id="d",
        file_name="blur.png",
        file_type="png",
        language="tr",
        pages=[PageContent(page_number=1, text="??", confidence=0.1)],
        success=True,
        element_count=1,
    )
    recovered = PageContent(page_number=1, text="Recovered clear page text.", source="qwen_vl")
    recovered._vision_meta = {}
    vision = MagicMock()
    vision.is_available.return_value = True
    vision.recover_pages.return_value = ([recovered], [], VisionInfo())

    with tempfile.TemporaryDirectory() as tmp:
        path = _png(Path(tmp) / "blur.png")
        with patch(
            "Agents.ocr_agent.processing.processor.PageImageExtractor.extract_pages",
            return_value=[(1, path)],
        ):
            result = OCRProcessor(
                OCRConfig(
                    quality=QualityConfig(min_text_length=40, min_confidence=0.5),
                    qwen=QwenVLConfig(enabled=True),
                ),
                ocr_provider=ocr,
                vision_provider=vision,
            ).process(path, document_id="blur", accompanying_text="yardımcı not")

    assert result.qwen_used is True
    kwargs = vision.recover_pages.call_args.kwargs
    assert kwargs.get("accompanying_text") == "yardımcı not"
    print("OK quality_fail_calls_qwen")


def test_qwen_unavailable_keeps_unstructured_result():
    from Agents.ocr_agent.config import OCRConfig, QwenVLConfig, QualityConfig
    from Agents.ocr_agent.models import NormalizedOCRResult, PageContent
    from Agents.ocr_agent.processing.processor import OCRProcessor

    # Weak but non-empty Unstructured result
    weak_text = "kisa metin"
    ocr = MagicMock()
    ocr.process.return_value = NormalizedOCRResult(
        document_id="d",
        file_name="w.png",
        file_type="png",
        language="tr",
        pages=[PageContent(page_number=1, text=weak_text, confidence=0.2)],
        success=True,
        element_count=2,
    )
    vision = MagicMock()
    vision.is_available.return_value = False

    with tempfile.TemporaryDirectory() as tmp:
        path = _png(Path(tmp) / "w.png")
        result = OCRProcessor(
            OCRConfig(
                quality=QualityConfig(min_text_length=100, min_confidence=0.9),
                qwen=QwenVLConfig(enabled=True),
            ),
            ocr_provider=ocr,
            vision_provider=vision,
        ).process(path, document_id="w")

    assert result.qwen_used is False
    assert weak_text in result.full_text
    vision.recover_pages.assert_not_called()
    print("OK qwen_unavailable_keeps_unstructured")


def test_unified_result_contract():
    payload = _good_unified().to_dict()
    assert set(payload.keys()) >= {
        "success",
        "document_info",
        "content",
        "vision",
        "classification",
    }
    assert payload["classification"] == {"document_type": None, "confidence": None}
    print("OK unified_contract")


if __name__ == "__main__":
    test_temp_file_handoff_and_cleanup_without_text()
    test_temp_file_with_accompanying_text()
    test_primary_ocr_success_skips_qwen()
    test_quality_fail_calls_qwen_with_text()
    test_qwen_unavailable_keeps_unstructured_result()
    test_unified_result_contract()
    print("All integration upload/OCR flow tests passed.")
