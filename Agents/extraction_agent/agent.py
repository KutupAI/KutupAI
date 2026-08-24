"""
extraction_agent -- extracts flat contact/document fields for Orchestration.

Pipeline: OCR -> Classification -> [THIS] -> Validation -> ...

Unified pipeline envelope contract (read / write):

  Input  (extraction empty):
    {request, ocr, classification, extraction: {}, validation,
     rag, summary, routing, writing}

  Output (same envelope, extraction filled):
    extraction: {
      "success": bool,
      "sender": str | null,
      "date": str | null,
      "address": str | null,
      "phone": str | null,
      "email": str | null
    }

Also writes Orchestration wire key extraction_result with the same contract
payload. Never calls Storage; never runs OCR itself.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from Agents.base.agent_registry import register
from Agents.base.base_agent import BaseAgent

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

# Canonical keys written to state["extraction"] (unified contract).
EXTRACTION_CONTRACT_KEYS = ("success", "sender", "date", "address", "phone", "email")


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _resolve_ocr_payload(state: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Prefer unified ``state["ocr"]["ocr_data"]``, then legacy ``ocr_result``."""
    ocr = _as_dict(state.get("ocr"))
    ocr_data = _as_dict(ocr.get("ocr_data"))
    if ocr_data.get("full_text") is not None or ocr_data.get("pages") is not None:
        text = str(ocr_data.get("full_text") or "")
        if not text.strip():
            pages = ocr_data.get("pages") or []
            text = "\n".join(
                str(p.get("text") or "") for p in pages if isinstance(p, dict)
            )
        vision = _as_dict(ocr_data.get("vision"))
        signature = _as_dict(vision.get("signature"))
        flags = {
            "has_signature": bool(signature.get("detected")),
            "has_handwritten_signature": bool(signature.get("handwritten")),
            "has_articles": False,
        }
        return text, flags

    ocr_legacy = state.get("ocr_result") or state.get("ocr_output") or {}
    if isinstance(ocr_legacy, str):
        return ocr_legacy, {}
    if not isinstance(ocr_legacy, dict):
        to_dict = getattr(ocr_legacy, "to_dict", None)
        if callable(to_dict):
            ocr_legacy = to_dict()
        else:
            return "", {}

    # Wire envelope: { Success, Data: [document, ...] }
    data = ocr_legacy.get("Data") or ocr_legacy.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        doc = data[0]
        text = str(doc.get("full_text") or "")
        vision = _as_dict(doc.get("vision"))
        signature = _as_dict(vision.get("signature"))
        return text, {
            "has_signature": bool(
                signature.get("detected") or doc.get("has_signature", False)
            ),
            "has_handwritten_signature": bool(
                signature.get("handwritten") or doc.get("has_handwritten_signature", False)
            ),
            "has_articles": bool(doc.get("has_articles", False)),
        }

    text = ocr_legacy.get("full_text") or "\n".join(ocr_legacy.get("lines", []) or [])
    return str(text or ""), {
        "has_signature": bool(ocr_legacy.get("has_signature", False)),
        "has_handwritten_signature": bool(ocr_legacy.get("has_handwritten_signature", False)),
        "has_articles": bool(ocr_legacy.get("has_articles", False)),
    }


def _get_classification_hint(state: Dict[str, Any]) -> Optional[str]:
    clf = _as_dict(state.get("classification")) or _as_dict(state.get("classification_result"))
    return clf.get("document_type") or clf.get("label") or clf.get("evrak_turu")


def _get_document_image_b64(state: Dict[str, Any]) -> Optional[str]:
    image = state.get("document_image_b64") or state.get("document_image")
    if isinstance(image, str) and image.strip():
        return image
    return None


def _first_field_value(values: list) -> Optional[str]:
    for fv in values:
        if getattr(fv, "value", None):
            return fv.value
    return None


def _extraction_contract(result: ExtractionResult) -> Dict[str, Any]:
    """Exact unified-contract shape for state['extraction']."""
    return {
        "success": bool(result.meta.success),
        "sender": _first_field_value(result.entities.person.ad_soyad),
        "date": result.document.tarih.value,
        "address": _first_field_value(result.entities.person.adres),
        "phone": _first_field_value(result.entities.person.telefon),
        "email": _first_field_value(result.entities.person.eposta),
    }


def _merge_result(state: Dict[str, Any], result: ExtractionResult) -> Dict[str, Any]:
    contract = _extraction_contract(result)
    state["extraction"] = contract
    # Dual-key convention (same as classification / validation): unified short
    # key + Orchestration wire key carry the same contract payload.
    state["extraction_result"] = contract
    return state


@register
class ExtractionAgent(BaseAgent):
    """Hybrid (Regex + NER + LLM + optional Vision) information extraction."""

    name = "extraction_agent"
    description = (
        "OCR ciktisindaki ham metni analiz ederek yapilandirilmis, guvenilir "
        "JSON verisi (sender/date/address/phone/email) uretir."
    )

    def __init__(self, config: ExtractionAgentConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG
        self.regex = RegexExtractor(self.cfg)
        self.ner = NEREngine(self.cfg)
        self.llm = HybridSemanticExtractor(self.cfg)
        self.vision = VisionFieldExtractor(self.cfg)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestration / envelope entry point.

        Accepts the unified pipeline envelope (``request`` / ``ocr`` /
        ``classification``) or legacy GraphState wire keys. Writes only
        ``extraction`` (+ ``extraction_result`` mirror); all other envelope
        keys pass through.
        """
        if not isinstance(state, dict):
            raise TypeError("ExtractionAgent.run expects GraphState / envelope as a dict")

        updated = dict(state)
        errors: list[str] = []
        warnings: list[str] = []

        text, ocr_flags = _resolve_ocr_payload(state)
        if not text.strip():
            warnings.append("OCR metni bos ya da bulunamadi - extraction atlandi")
            result = ExtractionResult(
                meta=ExtractionMeta(success=False, warnings=warnings, low_confidence=True)
            )
            return _merge_result(updated, result)

        classification_hint = _get_classification_hint(state)

        regex_out = self.regex.extract_all(text)

        ner_out = self.ner.extract_entities(text)
        ner_used = self.cfg.ner.enabled and self.ner.load_error is None
        if self.cfg.ner.enabled and self.ner.load_error:
            warnings.append(self.ner.load_error)

        llm_out = self.llm.extract(text, classification_hint=classification_hint)
        if llm_out.get("error"):
            warnings.append(str(llm_out["error"]))

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
            warnings.append(
                "Imza/el yazisi tespit edildi ama gorsel state'te bulunamadi (document_image_b64)"
            )

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

        result = ExtractionResult(
            document=document,
            entities=entities,
            request=request,
            vision=vision_info,
            meta=meta,
        )
        return _merge_result(updated, result)

    def process(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """Standalone envelope entry (same contract as ``run``).

        Input : {request, ocr, classification, extraction: {}, …}
        Output: same envelope with ``extraction`` filled per contract.
        """
        return self.run(envelope)

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


def process(envelope: Dict[str, Any], agent: Optional[ExtractionAgent] = None) -> Dict[str, Any]:
    """Module-level envelope entry matching classification_agent.process style."""
    return (agent or ExtractionAgent()).process(envelope)
