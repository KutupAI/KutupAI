"""Demo ExtractionAgent input/output envelope (offline, LLM disabled).

From the repository root:
    python Tests/Agents/manual_test_extraction.py

Live Inference Gemma (requires Inference/llama_server on :8080):
    python Tests/Agents/manual_test_extraction.py --live
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Agents.extraction_agent import ExtractionAgent, process
from Agents.extraction_agent.config import (
    ExtractionAgentConfig,
    LLMConfig,
    NERConfig,
    VisionConfig,
)

# Unified pipeline contract — extraction starts empty; classification filled.
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
    parser = argparse.ArgumentParser(description="Print extraction envelope result")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call real Inference LLM (llama-server). Default: offline (LLM off).",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.live:
        agent = ExtractionAgent(ExtractionAgentConfig.from_env())
    else:
        agent = ExtractionAgent(
            ExtractionAgentConfig(
                ner=NERConfig(enabled=False),
                llm=LLMConfig(enabled=False, use_langextract=False),
                vision=VisionConfig(enabled=False),
            )
        )

    output = process(SAMPLE_INPUT, agent=agent)
    print(_dump(_as_envelope(output)))


if __name__ == "__main__":
    main()
