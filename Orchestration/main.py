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

# Ensure project `.env` is visible to EvrenClient / agent configs.
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except Exception:
    pass

from fastapi import FastAPI, HTTPException

from Orchestration.process_service import run_workflow_from_application
from Orchestration.conversation_store import ConversationStore

logging.basicConfig(
    level=os.getenv("ORCHESTRATION_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def _configure_orchestration_logging() -> None:
    """Own the "Orchestration" logger instead of relying on the root config.

    PaddleX reconfigures the root logger while loading OCR weights, which
    silences every Orchestration.* record from that point on. A dedicated
    non-propagating handler survives that. Idempotent — safe to call again.
    """
    level = os.getenv("ORCHESTRATION_LOG_LEVEL", "INFO").upper()
    orchestration_logger = logging.getLogger("Orchestration")
    orchestration_logger.setLevel(level)
    orchestration_logger.propagate = False
    if not any(getattr(h, "_kutupai", False) for h in orchestration_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        handler._kutupai = True  # type: ignore[attr-defined]
        orchestration_logger.addHandler(handler)


_configure_orchestration_logging()
logger = logging.getLogger("Orchestration")

_OCR_READY = False


def _env_flag(name: str, default: bool = False) -> bool:
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
        _configure_orchestration_logging()  # PaddleX just reset the root logger.
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


def _warmup_rag_in_process() -> None:
    """Load the RAG embedding + reranker weights before serving traffic.

    Both are lru_cache singletons instantiated on first use, so without this
    the first /process request pays ~50s of model loading inside the RAG stage.
    """
    if not _env_flag("RAG_WARMUP_ON_STARTUP", True):
        logger.info("RAG warm-up skipped (RAG_WARMUP_ON_STARTUP=0)")
        return

    started = time.perf_counter()
    logger.info("RAG warm-up starting in-process…")
    try:
        from RAG.embeddings.embedding_model import embed_text

        embed_text("ısınma sorgusu")
        logger.info("RAG warm-up: embedding model ready (%.1fs)", time.perf_counter() - started)
    except Exception:
        logger.exception("RAG warm-up: embedding model failed — will load lazily")

    try:
        from RAG.configuration.rag_config_loader import reranker_config
        from RAG.retriever.reranker import _model

        if reranker_config.enabled:
            _model().predict([("ısınma sorgusu", "ısınma metni")])
        else:
            logger.info("RAG warm-up: reranker disabled in config")
    except Exception:
        logger.exception("RAG warm-up: reranker failed — will load lazily")

    logger.info(
        "RAG warm-up done in %.1fs — first request skips model loading",
        time.perf_counter() - started,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _warmup_ocr_in_process()
    _warmup_rag_in_process()
    yield


app = FastAPI(
    title="SmartGovernmentAI Orchestration",
    version="0.1.0",
    lifespan=lifespan,
)
conversation_store = ConversationStore()


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


@app.get("/conversations")
def list_conversations() -> dict[str, Any]:
    """Presentation Sidebar için kalıcı sohbet listesi."""
    return {"items": conversation_store.list_conversations()}


@app.get("/conversations/{chat_id}")
def get_conversation(chat_id: str) -> dict[str, Any]:
    conversation = conversation_store.get_conversation(chat_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/conversations/{chat_id}")
def delete_conversation(chat_id: str) -> dict[str, bool]:
    if not conversation_store.delete_conversation(chat_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"success": True}


def main() -> None:
    import uvicorn

    host = os.getenv("ORCHESTRATION_HOST", "127.0.0.1")
    port = int(os.getenv("ORCHESTRATION_PORT", "8000"))
    # log_config=None keeps the basicConfig above; uvicorn's default dictConfig
    # sets disable_existing_loggers and silences every Orchestration.* logger.
    uvicorn.run(
        "Orchestration.main:app",
        host=host,
        port=port,
        reload=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
