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
from Agents.classification_agent.taxonomy import DOCUMENT_CLASSES, VALID_CODES

# Real-world folder names -> taxonomy code. Keys are normalized (lowercased,
# stripped) before lookup, so trailing spaces / casing in folder names
# (e.g. "onay belgesi " with a trailing space) don't break the match.
FOLDER_NAME_TO_CODE: dict[str, str] = {
    "dilekçe": "dilekce",
    "başvuru belgesi": "basvuru_belgesi",
    "talep yazısı": "talep_yazisi",
    "şikayet": "sikayet_basvurusu",
    "şikâyet başvurusu": "sikayet_basvurusu",
    "itiraz başvurusu": "itiraz_basvurusu",
    "bilgi edinme başvurusu": "bilgi_edinme_basvurusu",
    "resmi yazı": "resmi_yazi",
    "üst yazı": "ust_yazi",
    "izin belgesi": "izin_belgesi",
    "onay belgesi": "onay_belgesi",
    "tutanak": "tutanak",
    "form": "form",
    "beyan - beyanname": "beyan_beyanname",
    "beyan": "beyan_beyanname",
    "bildirim - tebligat": "bildirim_tebligat",
    "rapor": "rapor",
    "kararlar": "karar_karar_yazisi",
    "karar": "karar_karar_yazisi",
    "sözleşme - protokol": "sozlesme_protokol",
    "sözleşme": "sozlesme_protokol",
    "diğer": "diger_belirsiz",
}

_DOCUMENT_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}

# Filenames matching these patterns are AI-generated/synthetic images, not
# real document scans/photos -- auto-flagged so §6's "synthetic docs never
# go in the test split" rule applies without a human re-tagging every row.
_SYNTHETIC_FILENAME_MARKERS = ("generated_image", "code_generated")


def normalize_folder_name(name: str) -> str:
    return name.strip().lower()


def folder_to_taxonomy_code(folder_name: str) -> str | None:
    return FOLDER_NAME_TO_CODE.get(normalize_folder_name(folder_name))


def discover_from_class_folders(root_dir: str | Path) -> list[LabeledDocument]:
    """Read a `<class_folder>/<file>` layout -- the label comes from the
    folder name itself, per the real dataset structure the team collected
    (one subfolder per taxonomy class). No manual labeling needed for this
    layout; only synthetic-vs-real still needs the filename heuristic
    below, since that can't be inferred from the folder.
    """
    root_dir = Path(root_dir)
    records: list[LabeledDocument] = []
    unmapped_folders: set[str] = set()

    for folder in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        code = folder_to_taxonomy_code(folder.name)
        if code is None:
            unmapped_folders.add(folder.name)
            continue

        for file_path in sorted(folder.iterdir()):
            if not file_path.is_file() or file_path.suffix.lower() not in _DOCUMENT_EXTENSIONS:
                continue
            is_synthetic = any(marker in file_path.name.lower() for marker in _SYNTHETIC_FILENAME_MARKERS)
            document_id = f"{code}__{file_path.stem}"
            records.append(
                LabeledDocument(
                    document_id=document_id,
                    pdf_path=str(file_path),
                    ocr_json_path=None,
                    label=code,
                    is_synthetic=is_synthetic,
                )
            )

    if unmapped_folders:
        print(f"[loader] WARNING: {len(unmapped_folders)} folder(s) did not match any taxonomy class and were skipped: {sorted(unmapped_folders)}")
        print("  -> add them to FOLDER_NAME_TO_CODE in loader.py if they should map to an existing class.")

    return records


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
    parser.add_argument("--pdf-dir", default=None, help="Flat folder of PDFs (unlabeled) -- old flow.")
    parser.add_argument("--ocr-dir", default=None)
    parser.add_argument(
        "--class-folders-dir",
        default=None,
        help="Folder containing one subfolder per class (e.g. 'dilekçe/', 'form/', ...). "
        "Label is taken from the folder name automatically -- see FOLDER_NAME_TO_CODE.",
    )
    parser.add_argument(
        "--output",
        default="Agents/classification_agent/dataset/manifest_template.csv",
    )
    args = parser.parse_args()

    if args.class_folders_dir:
        records = discover_from_class_folders(args.class_folders_dir)
    elif args.pdf_dir:
        records = discover_pairs(args.pdf_dir, args.ocr_dir)
    else:
        parser.error("Provide either --class-folders-dir or --pdf-dir.")
        return

    out = write_manifest_template(records, args.output)

    n_labeled = sum(1 for r in records if r.label)
    n_synthetic = sum(1 for r in records if r.is_synthetic)
    print(f"Found {len(records)} document(s) ({n_labeled} labeled, {n_synthetic} flagged synthetic). Manifest written to: {out}")
    if not args.class_folders_dir:
        print("Fill the 'label' column with one of these codes, then re-run distribution.py / splitter.py:")
        print(json.dumps(sorted(VALID_CODES), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()