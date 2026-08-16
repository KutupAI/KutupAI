"""
Unified processing contract forwarded by Orchestration.

Wire format (OCR Agent → Orchestration → Application → Presentation):
  { "Success": bool, "Data": [ document, ... ] }

The document object is produced by OCR Agent; this module is the
Orchestration-side pass-through of the same schema.
"""

from Agents.ocr_agent.models import (
    DOCUMENT_CONTRACT_KEYS,
    contract_envelope,
    empty_document,
    is_contract_envelope,
    normalize_document,
)

__all__ = [
    "DOCUMENT_CONTRACT_KEYS",
    "contract_envelope",
    "empty_document",
    "is_contract_envelope",
    "normalize_document",
]
