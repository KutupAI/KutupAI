"""
Orchestration POST /process wiring tests (OCRAgent mocked via workflow overrides).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_process_file_without_text():
    from Agents.ocr_agent.models import contract_envelope, empty_document
    from Orchestration.graph.graph_definition import Stage
    from Orchestration.process_service import run_workflow

    doc = empty_document(document_id="d1", file_name="a.png", file_type="png")
    doc["full_text"] = "OCR text"
    doc["pages"] = [
        {
            "page_number": 1,
            "text": "OCR text",
            "tables": [],
            "vision": {
                "signature": {"detected": False, "handwritten": False},
                "stamp": {"detected": False},
            },
        }
    ]
    agent = MagicMock()
    agent.run.side_effect = lambda state: {
        **state,
        "ocr_status": "completed",
        "ocr_result": contract_envelope(True, [doc]),
        "document_text": "OCR text",
    }

    with __import__("tempfile").TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        out = run_workflow(
            document_id="d1",
            document_path=str(path),
            accompanying_text=None,
            agent_overrides={Stage.OCR: agent},
        )

    assert set(out.keys()) == {"Success", "Data"}
    assert out["Success"] is True
    assert out["Data"][0]["full_text"] == "OCR text"
    call_state = agent.run.call_args[0][0]
    assert call_state["document_path"] == str(path)
    print("OK process_file_without_text")


def test_process_file_with_text():
    from Agents.ocr_agent.models import contract_envelope, empty_document
    from Orchestration.graph.graph_definition import Stage
    from Orchestration.process_service import run_workflow

    agent = MagicMock()
    doc = empty_document(
        document_id="d2",
        file_name="b.pdf",
        file_type="pdf",
        question="",
    )
    doc["full_text"] = "ok"
    doc["pages"] = [
        {
            "page_number": 1,
            "text": "ok",
            "tables": [],
            "vision": {
                "signature": {"detected": False, "handwritten": False},
                "stamp": {"detected": False},
            },
        }
    ]
    agent.run.side_effect = lambda state: {
        **state,
        "ocr_status": "completed",
        "ocr_result": contract_envelope(True, [doc]),
    }

    with __import__("tempfile").TemporaryDirectory() as tmp:
        path = Path(tmp) / "b.pdf"
        path.write_bytes(b"%PDF-1.4")
        out = run_workflow(
            document_id="d2",
            document_path=str(path),
            accompanying_text="lütfen imzayı kontrol et",
            agent_overrides={Stage.OCR: agent},
        )

    assert out["Success"] is True
    assert out["Data"][0]["question"] == "lütfen imzayı kontrol et"
    state = agent.run.call_args[0][0]
    assert state.get("accompanying_text") == "lütfen imzayı kontrol et" or state.get("question")
    print("OK process_file_with_text")


def test_process_missing_path():
    from Orchestration.process_service import run_workflow

    out = run_workflow(document_id="x", document_path=None)
    assert set(out.keys()) == {"Success", "Data"}
    assert out["Success"] is False
    assert out["Data"][0]["document_id"] == "x"
    print("OK process_missing_path")


if __name__ == "__main__":
    test_process_file_without_text()
    test_process_file_with_text()
    test_process_missing_path()
    print("All Orchestration process tests passed.")
