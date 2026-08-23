"""
Standalone OCR Agent test runner.

Runs the exact contract used with Orchestration, without booting the rest
of the project:

    Storage file -> Unified Input (state["request"]["document"]) ->
    OCRAgent().run(state) -> Unified Output (state["ocr"])

Usage
-----
    # Full pipeline on a real file from Storage:
    python Tests/Agents/test_ocr_agent_standalone.py --file "Storage/files/uploads/Elektrik sozlesmesi.pdf"

    # Same, but with a document_id/file_name pair resolved through Storage
    # (see Agents/ocr_agent/agent.py::_resolve_via_storage):
    python Tests/Agents/test_ocr_agent_standalone.py --document-id DOC-001 --file-name "Elektrik sozlesmesi.pdf"

    # Force PaddleOCR-VL vision fallback to confirm the local llama-server
    # on port 8111 is actually reachable and responding, independent of
    # whether the plain OCR pass would have needed it:
    python Tests/Agents/test_ocr_agent_standalone.py --file "Storage/files/uploads/scan.pdf" --force-vision-fallback

Notes
-----
- This must be run from (or with PYTHONPATH set to) the project root, i.e.
  the directory that contains `Agents/`, so `import Agents.ocr_agent` works.
  This script also inserts that root onto sys.path automatically.
- `--force-vision-fallback` does NOT run the full adaptive pipeline; it
  renders the document's first page and calls the configured vision
  fallback provider (PaddleOCR-VL via llama-server) directly, so a
  well-OCR'd test document still proves connectivity.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# --- make `Agents.ocr_agent` importable when run directly ------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # KutupAI/Tests/Agents/.. .. -> KutupAI/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ocr_agent.standalone_test")


def build_unified_state(document_id: str, file_name: str, file_type: str, question: str) -> dict:
    """Same unified state shape Orchestration hands to every Agent."""
    return {
        "request": {
            "success": True,
            "question": question,
            "document": {
                "document_id": document_id,
                "file_name": file_name,
                "file_type": file_type,
            },
        },
        "ocr": {},
        "classification": {},
        "extraction": {},
        "validation": {},
        "rag": {},
        "summary": {},
        "routing": {},
        "writing": {},
    }


def run_full_pipeline(args: argparse.Namespace) -> dict:
    from Agents.ocr_agent import OCRAgent

    if args.file:
        path = Path(args.file)
        file_name = path.name
        document_id = args.document_id or path.stem
    else:
        if not args.file_name:
            raise SystemExit("Provide either --file, or --document-id + --file-name.")
        file_name = args.file_name
        document_id = args.document_id or Path(file_name).stem

    file_type = Path(file_name).suffix.lstrip(".")
    state = build_unified_state(document_id, file_name, file_type, args.question)

    # Backward-compatible direct path (agent.py checks this before Storage
    # resolution), so a locally-given --file always works without a real
    # Storage module being importable.
    if args.file:
        state["document_path"] = str(Path(args.file))

    logger.info("Unified INPUT state:\n%s", json.dumps(state, ensure_ascii=False, indent=2))

    agent = OCRAgent()
    result_state = agent.run(state)

    logger.info(
        "Unified OUTPUT state[\"ocr\"]:\n%s",
        json.dumps(result_state.get("ocr"), ensure_ascii=False, indent=2)[:4000],
    )
    logger.info(
        "Wire OUTPUT state[\"ocr_result\"] Success=%s pages=%s",
        (result_state.get("ocr_result") or {}).get("Success"),
        len(((result_state.get("ocr_result") or {}).get("Data") or [{}])[0].get("pages") or []),
    )

    # Sanity: every other section must be untouched.
    for section in ("classification", "extraction", "validation", "rag", "summary", "routing", "writing"):
        assert result_state.get(section) == {}, f"section '{section}' was modified unexpectedly!"
    assert "ocr_result" in result_state, "Orchestration wire key ocr_result missing"
    assert "ocr_status" in result_state, "Orchestration wire key ocr_status missing"
    ocr = result_state.get("ocr") or {}
    if ocr.get("success"):
        data = ocr.get("ocr_data") or {}
        pages = data.get("pages") or []
        assert data.get("page_count", 0) == len(pages), "page_count must match pages[] length"
        assert pages, "successful OCR must populate ocr_data.pages (not empty)"
        assert any((p.get("text") or "").strip() for p in pages), "pages[].text must not all be empty"
    logger.info("OK: all non-OCR sections untouched; wire keys present.")
    return result_state


def run_force_vision_fallback(args: argparse.Namespace) -> None:
    """Bypass the adaptive OCR pipeline and call PaddleOCR-VL directly."""
    import cv2
    import numpy as np

    from Agents.ocr_agent.config import OCRConfig
    from Agents.ocr_agent.interfaces.vision_fallback import build_vision_fallback

    if not args.file:
        raise SystemExit("--force-vision-fallback requires --file so a page can be rendered.")

    path = Path(args.file)
    config = OCRConfig.from_env()

    if path.suffix.lower() == ".pdf":
        from Agents.ocr_agent.processing.pdf_renderer import PDFRenderer

        renderer = PDFRenderer(config.pdf_dpi, config.max_pdf_pages)
        pages = renderer.render(path)
        if not pages:
            raise SystemExit(f"Could not rasterize any page from {path}")
        image = pages[0]
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not decode image {path}")

    logger.info("[OCR] Vision fallback triggered (forced by --force-vision-fallback)")
    fallback = build_vision_fallback(config.vision_fallback)
    logger.info(
        "Using provider=%s endpoint=%s model=%s",
        fallback.provider_name, config.vision_fallback.endpoint, config.vision_fallback.model_name,
    )
    result = fallback.read_page(image)

    logger.info("PaddleOCR-VL result: provider=%s signature=%s handwritten=%s stamp=%s",
                result.provider, result.signature_detected, result.signature_handwritten, result.stamp_detected)
    logger.info("PaddleOCR-VL recovered text (first 500 chars):\n%s", result.text[:500])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", help="Path to a file already on disk (e.g. under Storage/files/uploads/).")
    parser.add_argument("--document-id", default=None, help="document_id to put in the unified input.")
    parser.add_argument("--file-name", default=None, help="file_name to put in the unified input "
                                                            "(used with Storage resolution when --file is omitted).")
    parser.add_argument("--question", default="", help="request.question value (optional, passthrough only).")
    parser.add_argument("--force-vision-fallback", action="store_true",
                         help="Skip the adaptive pipeline and call PaddleOCR-VL directly on page 1, "
                              "to verify http://127.0.0.1:8111/v1/chat/completions is reachable.")
    args = parser.parse_args()

    if args.force_vision_fallback:
        run_force_vision_fallback(args)
        return

    result_state = run_full_pipeline(args)
    ocr = result_state.get("ocr") or {}
    print("\n=== SUMMARY ===")
    print("success:", ocr.get("success"))
    print("status:", ocr.get("status"))
    if not ocr.get("success"):
        print("error:", ocr.get("error"))
    else:
        data = ocr.get("ocr_data") or {}
        print("page_count:", data.get("page_count"))
        print("language:", data.get("language"))
        print("vision:", data.get("vision"))
        print("full_text (first 300 chars):", (data.get("full_text") or "")[:300])


if __name__ == "__main__":
    main()
