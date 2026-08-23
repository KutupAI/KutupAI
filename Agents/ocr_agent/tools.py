"""Thin helpers for direct OCRClient use (not the Orchestration state contract)."""

from __future__ import annotations

from typing import Any

from Agents.ocr_agent.client import OCRClient, OCRRequest
from Agents.ocr_agent.config import OCRConfig


def run_ocr(
    file_path: str,
    *,
    document_id: str | None = None,
    config: OCRConfig | None = None,
    client: OCRClient | None = None,
) -> dict[str, Any]:
    """Return the internal OCRProcessor result dict."""
    active = client or OCRClient(config)
    return active.process(OCRRequest(file_path, document_id))
