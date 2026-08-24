"""
schema.py
-----------
Canonical metadata schema for every indexed chunk / source file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


REQUIRED_CHUNK_FIELDS = (
    "chunk_id",
    "source_file",
    "source_type",
    "law_name",
    "article_number",
)


@dataclass
class SourceMetadata:
    """Sidecar / document-level metadata (*.meta.json)."""

    law_name: str
    law_number: str = "unknown"
    effective_date: str = "unknown"
    source_type: str = "unknown"
    language: str = "tr"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    document_category: str = ""
    authority_level: str = "unknown"
    content_role: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceMetadata":
        return cls(
            law_name=str(data.get("law_name", "unknown")),
            law_number=str(data.get("law_number", "unknown")),
            effective_date=str(data.get("effective_date", "unknown")),
            source_type=str(data.get("source_type", "unknown")),
            language=str(data.get("language", "tr")),
            tags=list(data.get("tags", [])),
            notes=str(data.get("notes", "")),
            document_category=str(data.get("document_category", "")),
            authority_level=str(data.get("authority_level", "unknown")),
            content_role=str(data.get("content_role", "")),
        )


@dataclass
class ChunkMetadata:
    chunk_id: str
    source_file: str
    source_type: str
    law_name: str
    article_number: str
    chunk_index: int = 0
    law_number: str = "unknown"
    effective_date: str = "unknown"
    language: str = "tr"
    page: Optional[int] = None

    def to_chroma_dict(self) -> Dict[str, Any]:
        """Scalar-only dict accepted by Chroma metadata."""
        raw = asdict(self)
        return {
            key: value
            for key, value in raw.items()
            if value is not None and isinstance(value, (str, int, float, bool))
        }


def validate_chunk_metadata(meta: Dict[str, Any]) -> List[str]:
    """
    Gerekli alanların eksik olup olmadığını kontrol eder.
    🚀 KutupAI uyumluluğu: 'article_number' veya 'article_no' alanlarından en az biri varsa geçerli sayılır.
    """
    missing = []
    for f in REQUIRED_CHUNK_FIELDS:
        if f == "article_number":
            # Hibrit filtreleme için article_no da kabul edilebilir
            if (f not in meta or meta[f] in ("", None)) and (meta.get("article_no") in ("", None)):
                missing.append(f)
        else:
            if f not in meta or meta[f] in ("", None):
                missing.append(f)
                
    return missing
