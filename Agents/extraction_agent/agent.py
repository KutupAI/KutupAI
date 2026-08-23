"""
--------
Information Extraction Agent - main class.

Contract (Agents/base/base_agent.py):
    class BaseAgent(ABC):
        name: str
        description: str
        def run(self, state: Dict[str, Any]) -> Dict[str, Any]: ...

Pipeline position (per the task report):
    OCR Agent -> Classification Agent -> [THIS AGENT] -> Validation Agent -> ...

Expected input in `state` (defensive - tries several common key names so
this plugs in regardless of exact key naming used by teammates):
    - OCR output   : state["ocr_result"] | state["ocr_output"]
                     -> either a dict shaped like UnifiedOCRResult.to_dict()
                        (has "full_text" / "lines") or a plain string.
    - Classification: state["classification_result"] | state["classification"]
                     -> dict with a "document_type" / "label" key (optional).
    - Image (opt.)  : state["document_image_b64"] - base64 PNG/JPEG, only used
                     when OCR flags has_signature/has_handwritten_signature or
                     a table was detected, per report section 7.

Output written to state:
    state["extraction_result"] = ExtractionResult.to_state_dict()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:  # real repo layout
    from Agents.base.agent_registry import register
    from Agents.base.base_agent import BaseAgent
except ImportError:  
    from abc import ABC, abstractmethod

    class BaseAgent(ABC):  # type: ignore[no-redef]
        name: str = "base_agent"
        description: str = ""

        @abstractmethod
        def run(self, state: Dict[str, Any]) -> Dict[str, Any]: ...

    def register(cls):  # type: ignore[no-redef]
        return cls

from .config import DEFAULT_CONFIG, ExtractionAgentConfig
from .models import (
    DocumentInfo,
    Entities,
    ExtractionMeta,
    ExtractionResult,
    FieldValue,
    OrganizationInfo,
    PersonInfo,
    SemanticInfo,
    VisionInfo,
)
from .tools import HybridSemanticExtractor, NEREngine, RegexExtractor, VisionFieldExtractor

logger = logging.getLogger("extraction_agent")


# Defensive state readers - tolerate key-naming drift between teammates
# 
def _get_ocr_text_and_flags(state: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    ocr = state.get("ocr_result") or state.get("ocr_output") or {}
    if isinstance(ocr, str):
        return ocr, {}
    if not isinstance(ocr, dict):
        # dataclass instance (UnifiedOCRResult) - try to_dict()
        to_dict = getattr(ocr, "to_dict", None)
        if callable(to_dict):
            ocr = to_dict()
        else:
            return "", {}

    text = ocr.get("full_text") or "\n".join(ocr.get("lines", []) or [])
    flags = {
        "has_signature": ocr.get("has_signature", False),
        "has_handwritten_signature": ocr.get("has_handwritten_signature", False),
        "has_articles": ocr.get("has_articles", False),
    }
    return text or "", flags


def _get_classification_hint(state: Dict[str, Any]) -> Optional[str]:
    clf = state.get("classification_result") or state.get("classification") or {}
    if isinstance(clf, dict):
        return clf.get("document_type") or clf.get("label") or clf.get("evrak_turu")
    return None


def _get_document_image_b64(state: Dict[str, Any]) -> Optional[str]:
    return state.get("document_image_b64") or state.get("document_image")


def _to_unified_contract(result: "ExtractionResult") -> dict[str, Any]:
    """Flatten ExtractionResult into the short-key contract validation_agent
    reads (state["extraction"]): {success, sender, date, address, phone,
    email}. This mirrors ocr_agent's own dual-key convention (state["ocr"]
    unified + state["ocr_result"] wire) -- extraction_agent previously only
    wrote the wire key (state["extraction_result"]), so validation_agent's
    EXTRACTION_FIELDS lookup always saw an empty dict regardless of how the
    run actually went.
    """

    def _first(values: list) -> Optional[str]:
        for fv in values:
            if fv.value:
                return fv.value
        return None

    return {
        "success": result.meta.success,
        "sender": _first(result.entities.person.ad_soyad),
        "date": result.document.tarih.value,
        "address": _first(result.entities.person.adres),
        "phone": _first(result.entities.person.telefon),
        "email": _first(result.entities.person.eposta),
    }


@register
class ExtractionAgent(BaseAgent):
    """Hybrid (Regex + NER + LLM + optional Vision) information extraction."""

    name = "extraction_agent"
    description = (
        "OCR ciktisindaki ham metni analiz ederek yapilandirilmis, guvenilir "
        "JSON verisi (evrak/kisi/kurum/anlamsal bilgiler) uretir."
    )

    def __init__(self, config: ExtractionAgentConfig = DEFAULT_CONFIG):
        self.cfg = config
        self.regex = RegexExtractor(config)
        self.ner = NEREngine(config)
        # HybridSemanticExtractor = LLMSemanticExtractor (request_type/topic/
        # intent/keywords, unchanged) + LangExtract-grounded persons/
        # organizations when config.llm.use_langextract is True. Same
        # .extract() contract either way, so nothing else below changes.
        self.llm = HybridSemanticExtractor(config)
        self.vision = VisionFieldExtractor(config)

    # ------------------------------------------------------------------
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []

        text, ocr_flags = _get_ocr_text_and_flags(state)
        if not text.strip():
            warnings.append("OCR metni bos ya da bulunamadi - extraction atlandi")
            result = ExtractionResult(meta=ExtractionMeta(success=False, warnings=warnings, low_confidence=True))
            state["extraction_result"] = result.to_state_dict()
            state["extraction"] = _to_unified_contract(result)
            return state

        classification_hint = _get_classification_hint(state)

        # 1) Rule-Based
        regex_out = self.regex.extract_all(text)

        # 2) NLP (NER) - devre disi varsayilan olarak (bkz. config.py NERConfig).
        # Aktif edilirse ayri bir model yuklenir; degilse sessizce bos doner.
        ner_out = self.ner.extract_entities(text)
        ner_used = self.cfg.ner.enabled and self.ner.load_error is None
        if self.cfg.ner.enabled and self.ner.load_error:
            warnings.append(self.ner.load_error)

        # 3) LLM semantic (persons/organizations da bu adimda LLM'den geliyor -
        # ayri bir NER modeli kurmadan Qwen-VL uzerinden kisi/kurum cikarimi).
        llm_out = self.llm.extract(text, classification_hint=classification_hint)
        if llm_out.get("error"):
            warnings.append(str(llm_out["error"]))

        # 4) Vision (only if signature/handwriting flagged and image provided)
        vision_info = VisionInfo()
        needs_vision = ocr_flags.get("has_signature") or ocr_flags.get("has_handwritten_signature")
        image_b64 = _get_document_image_b64(state)
        if needs_vision and image_b64:
            vis_out = self.vision.extract_from_image(image_b64)
            if vis_out.get("error"):
                warnings.append(str(vis_out["error"]))
            vdata = vis_out.get("data") or {}
            vision_info = VisionInfo(
                used=bool(vis_out.get("used")),
                has_signature=vdata.get("has_signature"),
                has_stamp=vdata.get("has_stamp"),
                has_table=vdata.get("has_table"),
                has_handwriting=vdata.get("has_handwriting"),
                form_fields=vdata.get("form_fields", {}) or {},
            )
        elif needs_vision and not image_b64:
            warnings.append("Imza/el yazisi tespit edildi ama gorsel state'te bulunamadi (document_image_b64)")

        # ---- merge into standard schema ----
        document = DocumentInfo(
            evrak_turu=classification_hint,
            tarih=regex_out["dates"][0] if regex_out["dates"] else FieldValue.empty(),
            sayi=regex_out["evrak_no"] or FieldValue.empty(),
            konu=FieldValue(
                value=(llm_out.get("data") or {}).get("topic"),
                confidence=float((llm_out.get("data") or {}).get("confidence", 0.0)),
                source="llm",
            ),
        )

        llm_data = llm_out.get("data") or {}
        llm_conf = float(llm_data.get("confidence", 0.0))
        langextract_used = bool(llm_out.get("langextract_used"))
        if llm_out.get("langextract_error"):
            warnings.append(str(llm_out["langextract_error"]))

        def _spanned_field_values(names: list[str], spans: list[Optional[dict]]) -> list[FieldValue]:
            values: list[FieldValue] = []
            for i, name in enumerate(names):
                if not name:
                    continue
                span = spans[i] if i < len(spans) else None
                # Grounded (LangExtract char-aligned) values get a higher,
                # verified confidence than the plain LLM guess -- the span
                # match itself is evidence the text literally contains this
                # value, not a paraphrase/hallucination.
                confidence = 0.95 if (langextract_used and span) else llm_conf
                values.append(
                    FieldValue(
                        value=name,
                        confidence=confidence,
                        source="llm",
                        char_start=(span or {}).get("start"),
                        char_end=(span or {}).get("end"),
                    )
                )
            return values

        llm_persons = _spanned_field_values(
            llm_data.get("persons") or [], llm_data.get("persons_spans") or []
        )
        llm_orgs = _spanned_field_values(
            llm_data.get("organizations") or [], llm_data.get("organizations_spans") or []
        )

        entities = Entities(
            person=PersonInfo(
                
                ad_soyad=ner_out.get("person", []) + llm_persons,
                telefon=regex_out["phones"],
                eposta=regex_out["emails"],
                adres=[],
            ),
            organization=OrganizationInfo(
                kurum=ner_out.get("organization", []) + llm_orgs,
                mudurluk=[],
                ilgili_birim=[],
            ),
        )
        request = SemanticInfo(
            request_type=llm_data.get("request_type"),
            topic=llm_data.get("topic"),
            intent=llm_data.get("intent"),
            keywords=llm_data.get("keywords", []) or [],
            missing_info=llm_data.get("missing_info", []) or [],
            confidence=float(llm_data.get("confidence", 0.0)),
        )

        overall_confidence = self._compute_overall_confidence(regex_out, ner_out, request)
        meta = ExtractionMeta(
            overall_confidence=overall_confidence,
            low_confidence=overall_confidence < self.cfg.confidence_threshold,
            retried=bool(llm_out.get("retried")),
            retry_count=int(llm_out.get("retry_count", 0)),
            llm_used=bool(llm_out.get("used")),
            langextract_used=langextract_used,
            ner_used=ner_used,
            vision_used=vision_info.used,
            errors=errors,
            warnings=warnings,
        )

        result = ExtractionResult(document=document, entities=entities, request=request, vision=vision_info, meta=meta)
        state["extraction_result"] = result.to_state_dict()
        # Unified short-key contract for validation_agent, mirroring the
        # same dual-key convention ocr_agent already uses
        # (state["ocr"] short/unified + state["ocr_result"] wire format).
        # extraction_result's nested schema (document/entities/request) was
        # never mirrored into a flat "extraction" key before this, so
        # validation_agent always read an empty {} and reported
        # extraction_result_missing even on successful runs.
        state["extraction"] = _to_unified_contract(result)
        return state

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_overall_confidence(regex_out: dict, ner_out: dict, request: SemanticInfo) -> float:
        scores: list[float] = [request.confidence] if request.confidence else []
        if regex_out.get("dates"):
            scores.append(0.9)
        if regex_out.get("evrak_no"):
            scores.append(regex_out["evrak_no"].confidence)
        for group in ner_out.values():
            scores.extend(fv.confidence for fv in group)
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 3)