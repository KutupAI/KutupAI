"""Stable OCR entry used by OCRAgent / Orchestration."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Agents.ocr_agent.config import OCRConfig
from Agents.ocr_agent.processing.processor import OCRProcessor

_PROCESSOR_CACHE: dict[OCRConfig, OCRProcessor] = {}
_PROCESSOR_CACHE_LOCK = threading.Lock()


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
        self.processor = processor or self._shared_processor(self.config)

    @staticmethod
    def _shared_processor(config: OCRConfig) -> OCRProcessor:
        with _PROCESSOR_CACHE_LOCK:
            proc = _PROCESSOR_CACHE.get(config)
            if proc is None:
                proc = OCRProcessor(config)
                _PROCESSOR_CACHE[config] = proc
            return proc

    def process(self, request: OCRRequest) -> dict[str, Any]:
        """Run OCR and return the stable output contract (see README)."""
        return self.processor.process(request.file_path, request.document_id)

    def process_file(
        self,
        file_path: str | Path,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        return self.process(OCRRequest(str(file_path), document_id))
