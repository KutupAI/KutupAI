"""
loader.py
-----------
Turns a folder of real PDFs (+ optional OCR JSON) into a labeling manifest,
and reads that manifest back once the team has filled in the `label`
column.

Intended flow once real data arrives (task doc section 6):

    python -m Agents.classification_agent.dataset.loader \
        --pdf-dir /path/to/pdfs --ocr-dir /path/to/ocr_json \
        --output Agents/classification_agent/dataset/manifest_template.csv

  -> open manifest_template.csv, fill the `label` column with one of the
     18 taxonomy codes (printed by this script), save.

    python -m Agents.classification_agent.dataset.distribution \
        --manifest manifest_template.csv

  -> class distribution table (section 6 deliverable).

    python -m Agents.classification_agent.dataset.splitter \
        --manifest manifest_template.csv --output-dir dataset/splits

  -> stratified train/val/test manifests (section 6/11 deliverable).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from Agents.classification_agent.dataset.schema import MANIFEST_COLUMNS, LabeledDocument
from Agents.classification_agent.taxonomy import VALID_CODES


def discover_pairs(pdf_dir: str | Path, ocr_json_dir: str | Path | None = None) -> list[LabeledDocument]:
    """Match every PDF in pdf_dir with an OCR JSON of the same stem in
    ocr_json_dir (if given). Unmatched PDFs are still included with
    ocr_json_path=None -- ocr_agent can be run on them later, or
    classification_agent can fall back to image-only input.
    """
    pdf_dir = Path(pdf_dir)
    ocr_json_dir = Path(ocr_json_dir) if ocr_json_dir else None

    records: list[LabeledDocument] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        ocr_path = None
        if ocr_json_dir:
            candidate = ocr_json_dir / f"{pdf_path.stem}.json"
            if candidate.exists():
                ocr_path = str(candidate)
        records.append(
            LabeledDocument(
                document_id=pdf_path.stem,
                pdf_path=str(pdf_path),
                ocr_json_path=ocr_path,
            )
        )
    return records


def write_manifest_template(records: list[LabeledDocument], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
    return output_path


def load_manifest(manifest_path: str | Path, *, require_labels: bool = False) -> list[LabeledDocument]:
    manifest_path = Path(manifest_path)
    records: list[LabeledDocument] = []
    invalid: list[tuple[str, str]] = []

    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = LabeledDocument.from_row(row)
            if record.label is not None and record.label not in VALID_CODES:
                invalid.append((record.document_id, record.label))
            records.append(record)

    if invalid:
        details = ", ".join(f"{doc_id}='{label}'" for doc_id, label in invalid)
        raise ValueError(
            f"Manifest has {len(invalid)} row(s) with a label outside taxonomy.py: {details}. "
            f"Valid codes: {sorted(VALID_CODES)}"
        )

    if require_labels:
        unlabeled = [r.document_id for r in records if not r.label]
        if unlabeled:
            raise ValueError(f"{len(unlabeled)} document(s) still have no label: {unlabeled[:10]}...")

    return records


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Discover PDF(+OCR JSON) pairs and write a labeling manifest.")
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--ocr-dir", default=None)
    parser.add_argument(
        "--output",
        default="Agents/classification_agent/dataset/manifest_template.csv",
    )
    args = parser.parse_args()

    records = discover_pairs(args.pdf_dir, args.ocr_dir)
    out = write_manifest_template(records, args.output)

    print(f"Found {len(records)} document(s). Manifest template written to: {out}")
    print("Fill the 'label' column with one of these codes, then re-run distribution.py / splitter.py:")
    print(json.dumps(sorted(VALID_CODES), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
