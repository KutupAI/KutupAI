"""OCR Agent package — Orchestration/tests public surface."""

from Agents.ocr_agent.agent import OCRAgent
from Agents.ocr_agent.client import OCRClient, OCRRequest
from Agents.ocr_agent.config import OCRConfig
from Agents.ocr_agent.models import (
    DOCUMENT_CONTRACT_KEYS,
    contract_envelope,
    empty_document,
    is_contract_envelope,
    normalize_document,
)
from Agents.ocr_agent.processing.processor import OCRProcessor

__all__ = [
    "OCRAgent",
    "OCRClient",
    "OCRRequest",
    "OCRConfig",
    "OCRProcessor",
    "DOCUMENT_CONTRACT_KEYS",
    "contract_envelope",
    "empty_document",
    "is_contract_envelope",
    "normalize_document",
]
