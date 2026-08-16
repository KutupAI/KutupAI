"""
Orchestration entry point.

Exposes POST /process for Application (ChatService → OrchestrationClient).
Runs OCRAgent in-process via existing OCRClient → OCRProcessor pipeline.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from Orchestration.process_service import run_ocr_pipeline

logging.basicConfig(
    level=os.getenv("ORCHESTRATION_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("Orchestration")

app = FastAPI(title="SmartGovernmentAI Orchestration", version="0.1.0")


class ProcessRequest(BaseModel):
    document_id: str = Field(..., description="Correlation id from Application")
    document_path: str | None = Field(
        default=None, description="Temporary file path on shared disk"
    )
    text: str | None = None
    question: str | None = None
    accompanying_text: str | None = None


def _resolve_accompanying_text(payload: ProcessRequest) -> str | None:
    for value in (payload.accompanying_text, payload.text, payload.question):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
def process(payload: ProcessRequest) -> dict[str, Any]:
    return run_ocr_pipeline(
        document_id=payload.document_id,
        document_path=payload.document_path,
        accompanying_text=_resolve_accompanying_text(payload),
    )


def main() -> None:
    import uvicorn

    host = os.getenv("ORCHESTRATION_HOST", "127.0.0.1")
    port = int(os.getenv("ORCHESTRATION_PORT", "8000"))
    uvicorn.run(
        "Orchestration.main:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
