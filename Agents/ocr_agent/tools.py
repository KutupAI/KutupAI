"""Tools exposed to OCRAgent (external integrations stay here)."""

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
    """Run OCR and return a plain dict for graph_state['ocr_result']."""
    active = client or OCRClient(config)
    return active.process(OCRRequest(file_path, document_id))
