"""
models.py
-----------
Output data structures for classification_agent.

Shape mirrors the strict JSON schema required by the task document, §7:

{
  "document_id": "DOC-001",
  "document_type": "dilekce",
  "confidence": 0.94,
  "alternatives": [
    {"type": "talep_yazisi", "confidence": 0.04},
    {"type": "basvuru_belgesi", "confidence": 0.02}
  ],
  "status": "success"
}

Agent must never return free-text explanations — always parse-able JSON
(§7: "Agent her durumda parse edilebilir JSON üretmeli. Serbest açıklama
döndürmemeli.").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

ClassificationStatus = Literal["success", "needs_review", "failed"]


@dataclass
class ClassificationAlternative:
    type: str
    confidence: float


@dataclass
class ClassificationResult:
    success: bool
    document_id: str | None
    document_type: str | None
    confidence: float
    alternatives: list[ClassificationAlternative] = field(default_factory=list)
    status: ClassificationStatus = "success"

    # Provenance — which stage produced the decision. Not required by §7's
    # minimal schema, but needed for the ablation/comparison tests in §10
    # (OCR-only vs image-only vs OCR+image vs +layout, per-source metrics).
    source: str = "unknown"  
    ocr_confidence: float | None = None
    processing_ms: float = 0.0
    error: str | None = None
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Export exactly the schema shape from the task document, plus
        provenance fields needed downstream (evaluation, needs_review queue).
        Field order matches §7's example for easy diffing against it.
        """
        raw = asdict(self)
        return {
            "document_id": raw.get("document_id"),
            "document_type": raw.get("document_type"),
            "confidence": round(float(raw.get("confidence") or 0.0), 4),
            "alternatives": [
                {"type": a["type"], "confidence": round(float(a["confidence"]), 4)}
                for a in (raw.get("alternatives") or [])
            ],
            "status": raw.get("status"),
            "success": bool(raw.get("success")),
            "source": raw.get("source"),
            "ocr_confidence": raw.get("ocr_confidence"),
            "processing_ms": raw.get("processing_ms"),
            "schema_version": raw.get("schema_version", "1.0"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "error": raw.get("error"),
        }
