from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class BoundingBox:
    points: list[list[float]]

    def as_xyxy(self) -> list[float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return [min(xs), min(ys), max(xs), max(ys)]


@dataclass
class OCRTextItem:
    text: str
    confidence: float
    bounding_box: BoundingBox
    page_index: int
    source: str = "paddleocr"
    corrected_text: str | None = None
    correction_applied: bool = False


@dataclass
class LayoutElement:
    element_type: str
    confidence: float
    bounding_box: BoundingBox
    page_index: int
    source: str = "pp-structurev3"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableCell:
    row: int
    column: int
    text: str
    confidence: float | None = None


@dataclass
class TableResult:
    page_index: int
    bounding_box: BoundingBox | None
    cells: list[TableCell] = field(default_factory=list)
    html: str | None = None
    markdown: str | None = None
    confidence: float | None = None


@dataclass
class VisualElement:
    element_type: str
    confidence: float
    bounding_box: BoundingBox
    page_index: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageResult:
    page_index: int
    width: int
    height: int
    text: str
    text_items: list[OCRTextItem] = field(default_factory=list)
    layout: list[LayoutElement] = field(default_factory=list)
    tables: list[TableResult] = field(default_factory=list)
    visual_elements: list[VisualElement] = field(default_factory=list)
    processing_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)


@dataclass
class OCRProcessingError:
    code: str
    message: str
    page_index: int | None = None
    recoverable: bool = False


@dataclass
class UnifiedOCRResult:
    success: bool
    document_id: str | None
    file_name: str
    file_type: str
    language: str
    pages: list[PageResult]
    full_text: str
    errors: list[OCRProcessingError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    processing_ms: float = 0.0
    engine: str = "PaddleOCR + PP-StructureV3"
    schema_version: str = "1.1"

    # Structured insights
    lines: list[str] = field(default_factory=list)
    has_signature: bool = False
    has_handwritten_signature: bool = False
    signature_names: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    primary_date: str | None = None
    has_articles: bool = False
    articles: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Professional export for UI/agents.

        Readability: prefer `lines` / `articles[].lines` arrays instead of
        scanning long strings full of escaped \\n characters in JSON.
        """
        raw = asdict(self)

        for page in raw.get("pages") or []:
            page_lines = page.get("lines") or []
            if not page_lines and page.get("text"):
                page_lines = [ln for ln in str(page["text"]).split("\n") if ln.strip()]
                page["lines"] = page_lines
            if page_lines:
                page["text"] = "\n".join(page_lines)

        lines = raw.get("lines") or []
        if not lines and raw.get("full_text"):
            lines = [ln for ln in str(raw["full_text"]).split("\n") if ln.strip()]
            raw["lines"] = lines
        if lines:
            raw["full_text"] = "\n".join(lines)

        return {
            "success": raw["success"],
            "schema_version": raw.get("schema_version", "1.1"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "document_id": raw.get("document_id"),
            "file_name": raw.get("file_name"),
            "file_type": raw.get("file_type"),
            "language": raw.get("language"),
            "engine": raw.get("engine"),
            "processing_ms": raw.get("processing_ms"),
            "summary": {
                "has_signature": bool(raw.get("has_signature")),
                "has_handwritten_signature": bool(raw.get("has_handwritten_signature")),
                "signature_names": raw.get("signature_names") or [],
                "primary_date": raw.get("primary_date"),
                "dates": raw.get("dates") or [],
                "has_articles": bool(raw.get("has_articles")),
                "article_count": len(raw.get("articles") or []),
                "line_count": len(lines),
            },
            "has_signature": bool(raw.get("has_signature")),
            "has_handwritten_signature": bool(raw.get("has_handwritten_signature")),
            "signature_names": raw.get("signature_names") or [],
            "primary_date": raw.get("primary_date"),
            "dates": raw.get("dates") or [],
            "has_articles": bool(raw.get("has_articles")),
            "articles": raw.get("articles") or [],
            "lines": lines,
            "full_text": raw.get("full_text") or "",
            "pages": raw.get("pages") or [],
            "errors": raw.get("errors") or [],
            "warnings": raw.get("warnings") or [],
        }
