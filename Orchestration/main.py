"""
Orchestration entry point.

Exposes POST /process for Application (ChatService → OrchestrationClient).
Accepts the unified Application layer envelope
  { request, ocr, classification, extraction, validation, rag, summary, routing, writing }
or the legacy flat payload { document_id, document_path, question, … }.
Runs the full workflow graph in-process; OCR is stage 1 like every other Agent.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI

from Orchestration.process_service import run_workflow_from_application

logging.basicConfig(
    level=os.getenv("ORCHESTRATION_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("Orchestration")

app = FastAPI(title="SmartGovernmentAI Orchestration", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
def process(payload: Dict[str, Any]) -> dict[str, Any]:
    """Full workflow graph (OCR → … → Writing). Disabled stages are skipped."""
    return run_workflow_from_application(payload or {})


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
