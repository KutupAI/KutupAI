"""
Orchestration entry point.

Exposes POST /process for Application (ChatService → OrchestrationClient).
Accepts the unified Application layer envelope
  { request, ocr, classification, extraction, validation, rag, summary, routing, writing }
or the legacy flat payload { document_id, document_path, question, … }.
Runs the full workflow graph in-process; OCR is stage 1 like every other Agent.

OCR models are warmed inside THIS process on startup so the first /process
request does not pay the multi-minute ``_ensure_paddle()`` GPU init cost.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
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

_OCR_READY = False


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _warmup_ocr_in_process() -> None:
    """Load PaddleOCR weights into this process (shared engine singleton)."""
    global _OCR_READY
    if not _env_flag("OCR_WARMUP_ON_STARTUP", True):
        logger.info("OCR warm-up skipped (OCR_WARMUP_ON_STARTUP=0)")
        return

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    started = time.perf_counter()
    logger.info(
        "OCR warm-up starting in-process (device=%s)…",
        os.getenv("OCR_DEVICE", "gpu"),
    )
    try:
        from Agents.ocr_agent.config import OCRConfig
        from Agents.ocr_agent.engines.paddle_engine import get_shared_engine

        cfg = OCRConfig.from_env()
        engine = get_shared_engine(cfg)
        engine._ensure_paddle()
        elapsed = time.perf_counter() - started
        _OCR_READY = True
        logger.info(
            "OCR warm-up done in %.1fs (engine=%s device=%s) — first request skips init",
            elapsed,
            engine.engine_name,
            engine._resolved_device,
        )
    except Exception:
        elapsed = time.perf_counter() - started
        logger.exception(
            "OCR warm-up failed after %.1fs — first OCR request will init lazily",
            elapsed,
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _warmup_ocr_in_process()
    yield


app = FastAPI(
    title="SmartGovernmentAI Orchestration",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "ocr_ready": "true" if _OCR_READY else "false",
    }


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
