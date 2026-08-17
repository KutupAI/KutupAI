"""
Real OCR test runner.

Modes:

1) Process ALL supported files:
    python Tests\\Agents\\test_ocr_real.py

2) Process ONE file:
    python Tests\\Agents\\test_ocr_real.py document.pdf

Input:
    Storage/files/uploads/

Output:
    Storage/files/ocr_results/<filename>.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT OCR AGENT
# ============================================================

from Agents.ocr_agent import OCRAgent


# ============================================================
# PATHS
# ============================================================

UPLOADS_DIR = (
    PROJECT_ROOT
    / "Storage"
    / "files"
    / "uploads"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "Storage"
    / "files"
    / "ocr_results"
)


# ============================================================
# SUPPORTED FILES
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
}


# ============================================================
# HELPERS
# ============================================================

def find_files() -> list[Path]:
    """Return all supported files from uploads directory."""

    if not UPLOADS_DIR.exists():
        raise FileNotFoundError(
            f"Uploads directory does not exist:\n{UPLOADS_DIR}"
        )

    files = [
        path
        for path in sorted(UPLOADS_DIR.iterdir())
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    ]

    return files


def resolve_files() -> list[Path]:
    """
    Resolve input mode.

    No argument:
        process all files.

    Argument:
        process only requested file.
    """

    # --------------------------------------------------------
    # ALL FILES
    # --------------------------------------------------------

    if len(sys.argv) == 1:
        return find_files()

    # --------------------------------------------------------
    # SPECIFIC FILE
    # --------------------------------------------------------

    requested = Path(sys.argv[1])

    if not requested.is_absolute():
        requested = UPLOADS_DIR / requested

    requested = requested.resolve()

    if not requested.exists():
        raise FileNotFoundError(
            f"File does not exist:\n{requested}"
        )

    if not requested.is_file():
        raise ValueError(
            f"Path is not a file:\n{requested}"
        )

    if requested.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {requested.suffix}"
        )

    return [requested]


def save_json(path: Path, result: dict) -> None:
    """Save JSON using UTF-8 and preserve Turkish characters."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )


def get_output_path(input_file: Path) -> Path:
    """Return JSON result path."""

    return RESULTS_DIR / f"{input_file.stem}.json"


# ============================================================
# RESULT INFORMATION
# ============================================================

def extract_stats(result: dict) -> dict:
    """Extract useful statistics without modifying OCR result."""

    data = result.get("data") or {}

    pages = data.get("pages") or []

    full_text = data.get("full_text") or ""

    language = data.get("language") or {}

    processing = data.get("processing") or {}

    signatures = 0
    stamps = 0
    tables = 0

    for page in pages:

        tables += len(
            page.get("tables") or []
        )

        vision = page.get("vision") or {}

        signature = (
            vision.get("signature") or {}
        )

        stamp = (
            vision.get("stamp") or {}
        )

        if signature.get("detected") is True:
            signatures += 1

        if stamp.get("detected") is True:
            stamps += 1

    return {
        "pages": data.get(
            "page_count",
            len(pages),
        ),
        "language": language.get(
            "detected"
        ),
        "confidence": language.get(
            "confidence"
        ),
        "characters": len(full_text),
        "signatures": signatures,
        "stamps": stamps,
        "tables": tables,
        "engine": processing.get(
            "engine"
        ),
        "fallback_used": processing.get(
            "fallback_used"
        ),
        "processing_ms": processing.get(
            "processing_ms"
        ),
    }


# ============================================================
# TERMINAL OUTPUT
# ============================================================

def print_result(
    input_file: Path,
    output_file: Path,
    result: dict,
    index: int,
    total: int,
) -> None:

    stats = extract_stats(result)

    print()
    print("=" * 72)
    print(f"OCR FILE {index}/{total}")
    print("=" * 72)

    print(f"File        : {input_file.name}")
    print(f"Success     : {result.get('success')}")
    print(f"Status      : {result.get('status')}")

    print()
    print("Document:")
    print(f"  Pages       : {stats['pages']}")
    print(f"  Language    : {stats['language']}")
    print(f"  Confidence  : {stats['confidence']}")
    print(f"  Characters  : {stats['characters']}")

    print()
    print("Vision:")
    print(f"  Signatures  : {stats['signatures']}")
    print(f"  Stamps      : {stats['stamps']}")
    print(f"  Tables      : {stats['tables']}")

    print()
    print("Engine:")
    print(f"  OCR Engine  : {stats['engine']}")
    print(
        f"  Qwen Fallback: "
        f"{stats['fallback_used']}"
    )
    print(
        f"  Time        : "
        f"{stats['processing_ms']} ms"
    )

    print()
    print(f"JSON         : {output_file}")

    if result.get("success") is not True:

        error = result.get("error") or {}

        print()
        print("ERROR")
        print(f"  Code       : {error.get('code')}")
        print(
            f"  Message    : "
            f"{error.get('message')}"
        )

    print("=" * 72)


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(
    input_file: Path,
    index: int,
    total: int,
) -> bool:

    output_file = get_output_path(
        input_file
    )

    document_id = input_file.stem

    print()
    print("-" * 72)
    print(
        f"Processing {index}/{total}: "
        f"{input_file.name}"
    )
    print("-" * 72)

    started = datetime.now()

    try:

        # ----------------------------------------------------
        # REAL OCR AGENT
        # ----------------------------------------------------

        agent = OCRAgent()

        state = {
            "document_id": document_id,
            "document_path": str(input_file),
            "file_path": str(input_file),
        }

        updated_state = agent.run(state)

        finished = datetime.now()

        # ----------------------------------------------------
        # GET OCR RESULT
        # ----------------------------------------------------

        result = updated_state.get(
            "ocr_result"
        )

        if not isinstance(result, dict):

            result = {
                "success": False,
                "status": "failed",
                "error": {
                    "code": "INVALID_OCR_RESULT",
                    "message": (
                        "OCR Agent did not return "
                        "a valid OCR result."
                    ),
                },
                "data": {
                    "document_id": document_id,
                    "file_name": input_file.name,
                    "file_type": (
                        input_file.suffix
                        .lower()
                        .lstrip(".")
                    ),
                    "page_count": 0,
                    "language": {
                        "detected": None,
                        "confidence": 0.0,
                    },
                    "pages": [],
                    "full_text": "",
                    "processing": {},
                },
            }

        # ----------------------------------------------------
        # ADD TEST METADATA
        # ----------------------------------------------------

        result = dict(result)

        data = result.get("data")

        if isinstance(data, dict):

            data = dict(data)

            processing = data.get(
                "processing"
            )

            if isinstance(processing, dict):

                processing = dict(
                    processing
                )

                processing[
                    "test_runner"
                ] = "test_ocr_real.py"

                processing[
                    "test_started_at"
                ] = started.isoformat()

                processing[
                    "test_finished_at"
                ] = finished.isoformat()

                data["processing"] = (
                    processing
                )

            result["data"] = data

        # ----------------------------------------------------
        # SAVE JSON
        # ----------------------------------------------------

        save_json(
            output_file,
            result,
        )

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        print_result(
            input_file=input_file,
            output_file=output_file,
            result=result,
            index=index,
            total=total,
        )

        return result.get(
            "success"
        ) is True

    except Exception as exc:

        print()
        print("=" * 72)
        print(
            f"OCR FAILED: "
            f"{input_file.name}"
        )
        print("=" * 72)
        print(str(exc))
        print("=" * 72)

        return False


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print()
    print("=" * 72)
    print("SMART GOVERNMENT AI - REAL OCR TEST")
    print("=" * 72)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        files = resolve_files()

    except Exception as exc:

        print()
        print("ERROR:")
        print(exc)
        return 1

    if not files:

        print()
        print(
            "No supported files found in:"
        )
        print(UPLOADS_DIR)
        return 1

    print()
    print(
        f"Files found: {len(files)}"
    )

    print(
        f"Input : {UPLOADS_DIR}"
    )

    print(
        f"Output: {RESULTS_DIR}"
    )

    successful = 0
    failed = 0

    # ========================================================
    # PROCESS ALL FILES
    # ========================================================

    for index, input_file in enumerate(
        files,
        start=1,
    ):

        success = process_file(
            input_file=input_file,
            index=index,
            total=len(files),
        )

        if success:
            successful += 1
        else:
            failed += 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("FINAL OCR TEST SUMMARY")
    print("=" * 72)

    print(f"Total      : {len(files)}")
    print(f"Successful : {successful}")
    print(f"Failed     : {failed}")

    print()
    print(
        f"Results saved to:"
    )
    print(RESULTS_DIR)

    print("=" * 72)

    return 0 if failed == 0 else 1


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    raise SystemExit(main())