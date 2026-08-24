"""Demo ClassificationAgent input/output envelope (offline, mocked VLM).

From the repository root:
    python Tests/Agents/manual_test_classification.py

Live Inference Gemma (requires Inference/llama_server on :8080):
    python Tests/Agents/manual_test_classification.py --live
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Agents.classification_agent import ClassificationAgent, process
from Agents.classification_agent.config import ClassificationConfig

# Unified pipeline contract — classification starts empty.
SAMPLE_INPUT = {
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

ENVELOPE_KEYS = (
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


def _as_envelope(state: dict) -> dict:
    """Keep only the unified contract sections for display."""
    return {k: state.get(k, {}) for k in ENVELOPE_KEYS}


def _dump(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print classification envelope result")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call real VLM (llama-server). Default: mocked offline demo.",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    agent = ClassificationAgent(
        ClassificationConfig(use_fast_classifier=False, needs_review_threshold=0.60)
    )

    if args.live:
        output = process(SAMPLE_INPUT, agent=agent)
    else:
        with patch(
            "Agents.classification_agent.agent.run_vlm_classification",
            return_value={
                "document_type": "Elektrik sozlesmesi",
                "confidence": 0.95,
                "alternatives": [],
            },
        ):
            output = process(SAMPLE_INPUT, agent=agent)

    print(_dump(_as_envelope(output)))

if __name__ == "__main__":
    main()
