"""Demo RAG layer full envelope input/output (Layers_contracts).

From the repository root:

    # Offline shape demo (mocked retrieve — no Chroma needed)
    python Tests/RAG/manual_test_rag.py

    # Live retrieve against local index (needs prior ingestion)
    python Tests/RAG/manual_test_rag.py --live

    # Custom question
    python Tests/RAG/manual_test_rag.py --live --question "CMK 100. madde tutuklama şartları nelerdir?"
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

from RAG.client import handle_rag_request
from RAG.retriever.query_router import QueryPlan

# Canonical Layers_contracts envelope (same shape as Tests/Agents/test_envelope_contract.py).
# RAG starts empty; prior layers are already filled.
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
                "duzenlenmis sozlesme metni. Abonelik, fatura ve "
                "tuketim sartlari bu sozlesmede yer alir."
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

# Expected keys inside a successful rag section (Layers_contracts).
RAG_SECTION_KEYS = {"success", "rag_data"}
RAG_DATA_KEYS = {"operation", "query", "results"}
RESULT_ITEM_KEYS = {
    "chunk_id",
    "law_number",
    "law_name",
    "article_no",
    "page_start",
    "page_end",
    "text",
    "score",
}


def _as_envelope(state: dict) -> dict:
    return {k: state.get(k, {}) for k in ENVELOPE_KEYS}


def _dump(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _mock_results(*_args, **_kwargs):
    return [
        {
            "id": "chunk-demo-1",
            "text": (
                "MADDE 1 - Bu Kanun, elektrik enerjisi tedariki ve abonelik "
                "sozlesmelerine iliskin genel hukumleri duzenler."
            ),
            "score": 0.91,
            "metadata": {
                "chunk_id": "chunk-demo-1",
                "law_number": "6446",
                "law_name": "Elektrik Piyasasi Kanunu",
                "article_no": "1",
                "page_start": 1,
                "page_end": 1,
            },
        },
        {
            "id": "chunk-demo-2",
            "text": (
                "MADDE 4 - Tedarikci, aboneye fatura ve tuketim bilgilerini "
                "acik ve anlasilir bicimde sunmakla yukumludur."
            ),
            "score": 0.84,
            "metadata": {
                "chunk_id": "chunk-demo-2",
                "law_number": "6446",
                "law_name": "Elektrik Piyasasi Kanunu",
                "article_no": "4",
                "page_start": 3,
                "page_end": 3,
            },
        },
    ]


def _assert_contract(input_state: dict, output_state: dict) -> list[str]:
    """Return human-readable contract check lines."""
    lines: list[str] = []
    for key in ENVELOPE_KEYS:
        if key == "rag":
            continue
        ok = output_state.get(key) == input_state.get(key)
        lines.append(f"  passthrough {key}: {'OK' if ok else 'FAIL'}")

    rag = output_state.get("rag") or {}
    lines.append(f"  rag.success is bool: {'OK' if isinstance(rag.get('success'), bool) else 'FAIL'}")
    data = rag.get("rag_data") if isinstance(rag.get("rag_data"), dict) else {}
    lines.append(f"  rag has rag_data (not top-level data/answer): {'OK' if data else 'FAIL'}")
    lines.append(
        f"  rag_data keys {sorted(RAG_DATA_KEYS)}: "
        f"{'OK' if RAG_DATA_KEYS <= set(data.keys()) else 'FAIL'}"
    )
    lines.append(
        f"  operation == 'retrieve': {'OK' if data.get('operation') == 'retrieve' else 'FAIL'}"
    )
    lines.append(
        f"  no LLM answer field: {'OK' if 'answer' not in data and 'answer' not in rag else 'FAIL'}"
    )
    results = data.get("results") if isinstance(data.get("results"), list) else None
    lines.append(f"  results is list: {'OK' if results is not None else 'FAIL'}")
    if results:
        item_keys = set(results[0].keys())
        lines.append(
            f"  result item keys >= {sorted(RESULT_ITEM_KEYS)}: "
            f"{'OK' if RESULT_ITEM_KEYS <= item_keys else 'FAIL got ' + str(sorted(item_keys))}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Print RAG layer full input/output envelope")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call real retrieve against local index. Default: mocked offline demo.",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Override request.question in the sample envelope.",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sample_input = json.loads(json.dumps(SAMPLE_INPUT))
    if args.question:
        sample_input["request"]["question"] = args.question

    print("=" * 72)
    print("RAG LAYER — INPUT (full envelope)")
    print("=" * 72)
    print(_dump(_as_envelope(sample_input)))

    if args.live:
        output = handle_rag_request(sample_input)
    else:
        with (
            patch("RAG.client.contract_adapter.retrieve", side_effect=_mock_results),
            patch(
                "RAG.client.contract_adapter.choose_query_plan",
                return_value=QueryPlan(
                    "semantic_fast", "vector", False, True, False, "manual demo"
                ),
            ),
        ):
            output = handle_rag_request(sample_input)

    print()
    print("=" * 72)
    print("RAG LAYER — OUTPUT (full envelope)")
    print("=" * 72)
    print(_dump(_as_envelope(output)))

    print()
    print("=" * 72)
   

if __name__ == "__main__":
    main()
