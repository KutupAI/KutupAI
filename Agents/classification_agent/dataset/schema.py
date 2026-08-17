"""
schema.py
-----------
Manifest record schema for the classification training/eval corpus.

One LabeledDocument = one physical document = one row in the manifest
CSV/JSON. This is intentionally a flat, spreadsheet-friendly shape so a
human can open the auto-generated template in Excel/Sheets, fill the
`label` column, and hand it back -- no code required for the labeling
step itself (task doc section 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

MANIFEST_COLUMNS: tuple[str, ...] = (
    "document_id",
    "pdf_path",
    "ocr_json_path",
    "label",
    "is_synthetic",
    "hard_case_tags",
    "split",
    "notes",
)


@dataclass
class LabeledDocument:
    document_id: str
    pdf_path: str
    ocr_json_path: str | None = None
    label: str | None = None  # must be a code from taxonomy.py once filled in
    is_synthetic: bool = False  # section 6: synthetic docs must stay OUT of the test split
    hard_case_tags: list[str] = field(default_factory=list)  # codes from evaluation/hard_cases.py
    split: str | None = None  # "train" | "val" | "test", set by splitter.py
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "pdf_path": self.pdf_path,
            "ocr_json_path": self.ocr_json_path or "",
            "label": self.label or "",
            "is_synthetic": "1" if self.is_synthetic else "0",
            "hard_case_tags": "|".join(self.hard_case_tags),
            "split": self.split or "",
            "notes": self.notes,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "LabeledDocument":
        return cls(
            document_id=row["document_id"],
            pdf_path=row["pdf_path"],
            ocr_json_path=row.get("ocr_json_path") or None,
            label=(row.get("label") or "").strip() or None,
            is_synthetic=str(row.get("is_synthetic", "0")).strip() in {"1", "true", "True"},
            hard_case_tags=[t for t in (row.get("hard_case_tags") or "").split("|") if t],
            split=(row.get("split") or "").strip() or None,
            notes=row.get("notes", ""),
        )
