"""
---------
Bilgi Çıkarma Aracısı için Standart JSON çıktı şeması.

Raporun gereksinimini karşılar (bölüm 9 - JSON Çıktı Standardı):
{ document: {}, entities: {}, request: {}, validation: {} }

`validation` burada değil, alt kademe Doğrulama Aracısı tarafından doldurulur.

Bu aracı şunları üretir: document, entities, request, meta (kendi iç
güven/hata ayıklama bilgisi, Doğrulama Aracısının yeniden deneme kararı için yararlı).

Her çıkarılan değer bir `FieldValue` içine sarılır, böylece alt kademe aracılar
her zaman *ne kadar* emin olduğumuzu ve *hangi* yöntemin onu ürettiğini bilirler
(regex / ner / llm / vision) - bu, Doğrulama Aracısının
Bir alana güvenip güvenmeyeceğine veya yeniden denemeyi tetikleyip tetiklemeyeceğine karar vermesini sağlar.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

SourceType = Literal["regex", "ner", "llm", "vision", "unknown"]


class FieldValue(BaseModel):
    """A single extracted value with provenance."""

    value: Optional[str] = None
    confidence: float = 0.0
    source: SourceType = "unknown"
    # Character-offset provenance into the *original* OCR full_text, only
    # populated when source == "llm" and the LangExtract path produced a
    # grounded (char-aligned) span for this value. None for regex/ner
    # (which don't track offsets today) and for ungrounded LLM output.
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    @classmethod
    def empty(cls) -> "FieldValue":
        return cls(value=None, confidence=0.0, source="unknown")


class DocumentInfo(BaseModel):
    """Evrak bilgileri (section 5.1)."""

    evrak_turu: Optional[str] = None  
    tarih: FieldValue = Field(default_factory=FieldValue.empty)
    sayi: FieldValue = Field(default_factory=FieldValue.empty)
    konu: FieldValue = Field(default_factory=FieldValue.empty)


class PersonInfo(BaseModel):
    """Kisisel bilgiler (section 5.2)."""

    ad_soyad: List[FieldValue] = Field(default_factory=list)
    telefon: List[FieldValue] = Field(default_factory=list)
    eposta: List[FieldValue] = Field(default_factory=list)
    adres: List[FieldValue] = Field(default_factory=list)


class OrganizationInfo(BaseModel):
    """Kurumsal bilgiler (section 5.3)."""

    kurum: List[FieldValue] = Field(default_factory=list)
    mudurluk: List[FieldValue] = Field(default_factory=list)
    ilgili_birim: List[FieldValue] = Field(default_factory=list)


class Entities(BaseModel):
    person: PersonInfo = Field(default_factory=PersonInfo)
    organization: OrganizationInfo = Field(default_factory=OrganizationInfo)


class SemanticInfo(BaseModel):
    """Anlamsal bilgiler (section 5.4) - LLM'nin ciktisi."""

    request_type: Optional[str] = None  # "Sikayet" | "Bilgi Talebi" | "Basvuru" | ...
    topic: Optional[str] = None
    intent: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class VisionInfo(BaseModel):
    """Qwen-VL cikti bilgileri (section 7) - sadece image varsa doldurulur."""

    used: bool = False
    has_signature: Optional[bool] = None
    has_stamp: Optional[bool] = None
    has_table: Optional[bool] = None
    has_handwriting: Optional[bool] = None
    form_fields: dict[str, Any] = Field(default_factory=dict)


class ExtractionMeta(BaseModel):
    """Internal diagnostics - consumed by Validation Agent (section 10)."""

    # Explicit run-level outcome, mirroring the success flag ocr_agent and
    # classification_agent already expose in their own state contracts.
    # False only when extraction could not run at all (e.g. no OCR text) --
    # NOT the same as low_confidence, which means it ran but is unsure.
    success: bool = True
    overall_confidence: float = 0.0
    low_confidence: bool = False
    retried: bool = False
    retry_count: int = 0
    llm_used: bool = False
    langextract_used: bool = False
    ner_used: bool = False
    vision_used: bool = False
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Top-level standard output merged into shared graph state."""

    document: DocumentInfo = Field(default_factory=DocumentInfo)
    entities: Entities = Field(default_factory=Entities)
    request: SemanticInfo = Field(default_factory=SemanticInfo)
    vision: VisionInfo = Field(default_factory=VisionInfo)
    meta: ExtractionMeta = Field(default_factory=ExtractionMeta)

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")