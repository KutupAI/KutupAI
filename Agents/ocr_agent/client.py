"""Stable OCR entry used by OCRAgent / Orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from Agents.ocr_agent.config import OCRConfig
from Agents.ocr_agent.models import UnifiedOCRResult
from Agents.ocr_agent.processing.processor import OCRProcessor


@dataclass(frozen=True)
class OCRRequest:
    file_path: str
    document_id: str | None = None


class OCRClient:
    def __init__(
        self,
        config: OCRConfig | None = None,
        processor: OCRProcessor | None = None,
    ) -> None:
        self.config = config or OCRConfig.from_env()
        self.processor = processor or OCRProcessor(self.config)

    def process(self, request: OCRRequest) -> UnifiedOCRResult:
        return self.processor.process(request.file_path, request.document_id)

    def process_file(
        self,
        file_path: str | Path,
        document_id: str | None = None,
    ) -> UnifiedOCRResult:
        return self.process(OCRRequest(str(file_path), document_id))
