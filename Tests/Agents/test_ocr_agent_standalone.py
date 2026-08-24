from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_unified_state(
    document_id: str,
    file_name: str,
    file_type: str,
    question: str,
) -> dict:
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


def validate_contract(state: dict) -> None:
    expected_sections = {
        "request",
        "ocr",
        "classification",
        "extraction",
        "validation",
        "rag",
        "summary",
        "routing",
        "writing",
    }

    assert set(state.keys()) == expected_sections, (
        f"Unexpected state keys: {set(state.keys()) - expected_sections}"
    )

    ocr = state["ocr"]

    assert isinstance(ocr, dict), "ocr must be a dictionary"

    if not ocr.get("success"):
        return

    ocr_data = ocr.get("ocr_data")

    assert isinstance(
        ocr_data,
        dict,
    ), "ocr.ocr_data must be a dictionary"

    assert "page_count" in ocr_data
    assert "language" in ocr_data
    assert "pages" in ocr_data
    assert "full_text" in ocr_data
    assert "vision" in ocr_data

    assert isinstance(
        ocr_data["pages"],
        list,
    ), "ocr_data.pages must be a list"

    assert (
        ocr_data["page_count"]
        == len(ocr_data["pages"])
    ), "page_count must match pages length"

    for section in (
        "classification",
        "extraction",
        "validation",
        "rag",
        "summary",
        "routing",
        "writing",
    ):
        assert (
            state[section] == {}
        ), f"{section} was modified by OCR Agent"


def run_test(args: argparse.Namespace) -> dict:
    from Agents.ocr_agent import OCRAgent

    if args.file:
        file_path = Path(args.file)

        if not file_path.exists():
            raise SystemExit(
                f"File does not exist: {file_path}"
            )

        file_name = file_path.name
        document_id = (
            args.document_id
            or file_path.stem
        )

    else:
        if not args.file_name:
            raise SystemExit(
                "Provide --file or --file-name."
            )

        file_name = args.file_name
        document_id = (
            args.document_id
            or Path(file_name).stem
        )

        file_path = None

    file_type = Path(file_name).suffix.lstrip(".").lower()

    state = build_unified_state(
        document_id=document_id,
        file_name=file_name,
        file_type=file_type,
        question=args.question,
    )


    print(
        "\nocr input :",
        json.dumps(
            state,
            ensure_ascii=False,
            indent=2,
        ),
    )

    result = OCRAgent().run(state)

    output = {
        "request": result.get("request"),
        "ocr": result.get("ocr"),
        "classification": result.get("classification"),
        "extraction": result.get("extraction"),
        "validation": result.get("validation"),
        "rag": result.get("rag"),
        "summary": result.get("summary"),
        "routing": result.get("routing"),
        "writing": result.get("writing"),
    }

    validate_contract(output)

    print(
        "\nocr output :",
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
    )

    print("\nOCR contract test: OK")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone OCR Agent contract test."
    )

    parser.add_argument(
        "--file",
        help="Path to the document.",
    )

    parser.add_argument(
        "--document-id",
        default=None,
        help="Document ID.",
    )

    parser.add_argument(
        "--file-name",
        default=None,
        help="File name when --file is not provided.",
    )

    parser.add_argument(
        "--question",
        default="",
        help="User question.",
    )

    parser.add_argument(
        "--force-vision-fallback",
        action="store_true",
        help=(
            "Run the OCR pipeline with vision fallback enabled."
        ),
    )

    args = parser.parse_args()

    if args.force_vision_fallback:
        raise SystemExit(
            "--force-vision-fallback is handled by the OCR Agent pipeline. "
            "Use the normal OCR test."
        )

    run_test(args)


if __name__ == "__main__":
    main()