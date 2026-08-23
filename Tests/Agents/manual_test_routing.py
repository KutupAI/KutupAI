"""Demo RoutingAgent input/output envelope (offline, no LLM).

From the repository root:
    python Tests/Agents/manual_test_routing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Agents.routing_agent import process

# Pipeline contract sample — same keys as Layers_contracts / README.
# Text fields are filled enough for a real department (placeholders alone fail).
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
                "Elektrik alım satım sözleşmesinin hukuki incelemesi talep "
                "edilmektedir. Sözleşme şartnamesinin mevzuata uygunluğu "
                "hakkında hukuki görüş bildirilmesi rica olunur."
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
        "rag_data": {
            "operation": "retrieve",
            "query": "elektrik sözleşmesi hukuki inceleme",
            "results": [],
        },
    },
    "summary": {
        "success": True,
        "rag_summary_text": (
            "Elektrik sözleşmesinin hukuki incelemesi ve mevzuat uyumu "
            "için hukuki görüş talebi."
        ),
    },
    "routing": {},
    "writing": {},
}


def _dump(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    output = process(SAMPLE_INPUT)

    print("\\\\input")
    print("routing  :", _dump(SAMPLE_INPUT))
    print()
    print("\\\\output")
    print("routing  :", _dump(output))


if __name__ == "__main__":
    main()
