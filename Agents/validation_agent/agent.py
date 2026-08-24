"""
Validation Agent.

Reads upstream OCR / Classification / Extraction from either the shared
pipeline envelope (state["ocr"], state["classification"], state["extraction"])
or Orchestration GraphState wire keys (state["ocr_result"], …), then
DYNAMICALLY computes state["validation"] (+ state["validation_result"]).

Nothing in this file hardcodes a fixed result - every field below is
derived from the incoming state at call time.

state["validation"] = state["validation_result"] = {
    "success": bool,        # computed: True iff no hard errors found
    "is_complete": bool,    # computed: True iff extraction succeeded AND
                             #           every extraction field is present
    "errors": [str, ...],   # computed: only populated when a real problem
                             #           is found in this specific input
    "warnings": [str, ...], # computed: same, for soft signals
}

All other state keys are preserved unchanged - this agent only ever writes
to state["validation"] and its Orchestration wire mirror
state["validation_result"].
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Agents.base.base_agent import BaseAgent
from Agents.base.agent_registry import register

from Agents.validation_agent import config
from Agents.validation_agent import tools


EXTRACTION_FIELDS = ("sender", "date", "address", "phone", "email")


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_field_value(values: Any) -> Optional[str]:
    """Pull the first non-empty FieldValue.value from a list (extraction_result)."""
    if not isinstance(values, list):
        return None
    for item in values:
        if isinstance(item, dict):
            value = item.get("value")
            if value is not None and value != "":
                return str(value)
        elif item is not None and item != "":
            return str(item)
    return None


def resolve_ocr(state: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize state['ocr'] or GraphState ocr_result / document_text."""
    ocr = _as_dict(state.get("ocr"))
    ocr_data = _as_dict(ocr.get("ocr_data"))
    if ocr_data.get("full_text") is not None or ocr.get("success") is not None:
        return ocr

    ocr_result = _as_dict(state.get("ocr_result"))
    full_text = ""
    page_count = 0
    language = None
    pages: list = []
    vision: Dict[str, Any] = {
        "signature": {"detected": False, "handwritten": False},
        "stamp": {"detected": False},
    }

    # Wire envelope: { Success, Data: [document, ...] }
    data = ocr_result.get("Data") or ocr_result.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        doc = data[0]
        full_text = str(doc.get("full_text") or "")
        pages = list(doc.get("pages") or [])
        page_count = int(doc.get("page_count") or len(pages) or (1 if full_text else 0))
        language = doc.get("language")
        if isinstance(language, dict):
            language = language.get("detected")
        if isinstance(doc.get("vision"), dict):
            vision = doc["vision"]
        success = bool(ocr_result.get("Success", ocr_result.get("success", bool(full_text.strip()))))
    elif "full_text" in ocr_result or "pages" in ocr_result:
        # Raw OCR data object (some adapters pass the document payload directly).
        full_text = str(ocr_result.get("full_text") or "")
        pages = list(ocr_result.get("pages") or [])
        page_count = int(ocr_result.get("page_count") or len(pages) or (1 if full_text else 0))
        language = ocr_result.get("language")
        if isinstance(language, dict):
            language = language.get("detected")
        if isinstance(ocr_result.get("vision"), dict):
            vision = ocr_result["vision"]
        success = bool(ocr_result.get("success", bool(full_text.strip())))
    else:
        full_text = str(state.get("document_text") or state.get("text") or "")
        success = bool(full_text.strip()) if full_text else False
        if not ocr_result and not full_text:
            return {}

    return {
        "success": success,
        "ocr_data": {
            "page_count": page_count or (1 if full_text.strip() else 0),
            "language": language,
            "pages": pages,
            "full_text": full_text,
            "vision": vision,
        },
    }


def resolve_classification(state: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize state['classification'] or classification_result."""
    classification = _as_dict(state.get("classification"))
    if classification.get("success") is not None or classification.get("document_type") is not None:
        # Prefer short key; fill confidence alias if only "confidence" exists.
        if (
            classification.get("classification_confidence") is None
            and classification.get("confidence") is not None
        ):
            classification = {
                **classification,
                "classification_confidence": classification.get("confidence"),
            }
        return classification

    cr = _as_dict(state.get("classification_result"))
    if not cr:
        return {}

    confidence = cr.get("classification_confidence", cr.get("confidence"))
    success = cr.get("success")
    if success is None:
        status = cr.get("status")
        if status == "failed":
            success = False
        elif status in ("success", "needs_review", "completed") or cr.get("document_type") or cr.get("doc_type"):
            success = True

    return {
        "success": success,
        "document_type": cr.get("document_type") or cr.get("doc_type"),
        "classification_confidence": confidence,
        "status": cr.get("status"),
    }


def resolve_extraction(state: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize state['extraction'] or nested extraction_result."""
    extraction = _as_dict(state.get("extraction"))
    if extraction.get("success") is not None or any(f in extraction for f in EXTRACTION_FIELDS):
        return extraction

    er = _as_dict(state.get("extraction_result"))
    if not er:
        return {}

    # Already-flat payload (or mock with top-level fields).
    if er.get("success") is not None or any(f in er for f in EXTRACTION_FIELDS):
        return {
            "success": er.get("success"),
            "sender": er.get("sender"),
            "date": er.get("date"),
            "address": er.get("address"),
            "phone": er.get("phone"),
            "email": er.get("email"),
        }

    # Nested ExtractionResult.to_state_dict() shape.
    meta = _as_dict(er.get("meta"))
    document = _as_dict(er.get("document"))
    entities = _as_dict(er.get("entities"))
    person = _as_dict(entities.get("person"))
    tarih = _as_dict(document.get("tarih"))

    success = meta.get("success")
    if success is None and (document or entities or meta):
        success = True

    return {
        "success": success,
        "sender": _first_field_value(person.get("ad_soyad")),
        "date": tarih.get("value"),
        "address": _first_field_value(person.get("adres")),
        "phone": _first_field_value(person.get("telefon")),
        "email": _first_field_value(person.get("eposta")),
    }


@register
class ValidationAgent(BaseAgent):
    """Deterministic, input-dependent validation against the shared-state contract."""

    name = "validation_agent"

    def run(self, state: dict) -> dict:
        try:
            errors: list[str] = []
            warnings: list[str] = []

            extraction = resolve_extraction(state)
            classification = resolve_classification(state)
            ocr = resolve_ocr(state)

            extraction_success = extraction.get("success")
            field_values = {f: extraction.get(f) for f in EXTRACTION_FIELDS}

            def present(v):
                return v is not None and v != ""

            if extraction_success is False:
                errors.append("extraction_failed")
            elif extraction_success is None:
                warnings.append("extraction_result_missing")
            else:
                present_fields = [f for f, v in field_values.items() if present(v)]
                missing_count = len(EXTRACTION_FIELDS) - len(present_fields)

                if present_fields and missing_count > 0:
                    warnings.append("partial_extraction_data")

                if present(field_values["date"]):
                    if not tools.validate_date_format(str(field_values["date"])):
                        errors.append("invalid_date_format")

                if present(field_values["email"]):
                    if not tools.validate_email_format(str(field_values["email"]), config.EMAIL_PATTERN):
                        errors.append("invalid_email_format")

                if present(field_values["phone"]):
                    if not tools.validate_phone_format(str(field_values["phone"]), config.PHONE_DIGIT_LENGTH):
                        errors.append("invalid_phone_format")

            if classification.get("success") is False:
                warnings.append("classification_failed")

            confidence = classification.get("classification_confidence")
            if confidence is not None and confidence < config.MIN_CLASSIFICATION_CONFIDENCE:
                warnings.append("low_classification_confidence")

            if ocr.get("success") is False:
                warnings.append("ocr_failed")
            else:
                ocr_data = ocr.get("ocr_data") or {}
                full_text = ocr_data.get("full_text")
                if ocr.get("success") is True and (full_text is None or str(full_text).strip() == ""):
                    warnings.append("empty_ocr_text")

            is_complete = (
                extraction_success is True
                and not errors
                and all(present(field_values[f]) for f in EXTRACTION_FIELDS)
            )

            payload = {
                "success": len(errors) == 0,
                "is_complete": is_complete,
                "errors": errors,
                "warnings": warnings,
            }
            # Dual-key convention: short envelope key + Orchestration wire key.
            state["validation"] = payload
            state["validation_result"] = payload
            return state

        except Exception as exc:  # noqa: BLE001
            payload = {
                "success": False,
                "is_complete": False,
                "errors": [f"validation_agent_internal_error:{exc}"],
                "warnings": [],
            }
            state["validation"] = payload
            state["validation_result"] = payload
            return state
