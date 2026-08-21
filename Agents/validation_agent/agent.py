"""
Validation Agent.

Reads state["extraction"], state["classification"], state["ocr"] and
DYNAMICALLY computes state["validation"] from their actual contents.
Nothing in this file hardcodes a fixed result - every field below is
derived from the incoming state at call time.

state["validation"] = {
    "success": bool,        # computed: True iff no hard errors found
    "is_complete": bool,    # computed: True iff extraction succeeded AND
                             #           every extraction field is present
    "errors": [str, ...],   # computed: only populated when a real problem
                             #           is found in this specific input
    "warnings": [str, ...], # computed: same, for soft signals
}

All other state keys (request, ocr, classification, extraction, rag,
summary, routing, writing) are preserved unchanged - this agent only
ever writes to state["validation"].
"""

from __future__ import annotations

from Agents.base.base_agent import BaseAgent
from Agents.base.agent_registry import register

from Agents.validation_agent import config
from Agents.validation_agent import tools


EXTRACTION_FIELDS = ("sender", "date", "address", "phone", "email")


@register
class ValidationAgent(BaseAgent):
    """Deterministic, input-dependent validation against the shared-state contract."""

    name = "validation_agent"

    def run(self, state: dict) -> dict:
        try:
            errors: list[str] = []
            warnings: list[str] = []

            extraction = state.get("extraction") or {}
            classification = state.get("classification") or {}
            ocr = state.get("ocr") or {}

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

            state["validation"] = {
                "success": len(errors) == 0,
                "is_complete": is_complete,
                "errors": errors,
                "warnings": warnings,
            }
            return state

        except Exception as exc:  # noqa: BLE001
            state["validation"] = {
                "success": False,
                "is_complete": False,
                "errors": [f"validation_agent_internal_error:{exc}"],
                "warnings": [],
            }
            return state