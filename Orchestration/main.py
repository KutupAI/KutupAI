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

from fastapi import FastAPI, HTTPException

from Orchestration.process_service import run_workflow_from_application
from Orchestration.conversation_store import ConversationStore

logging.basicConfig(
    level=os.getenv("ORCHESTRATION_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("Orchestration")

app = FastAPI(title="SmartGovernmentAI Orchestration", version="0.1.0")
conversation_store = ConversationStore()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
    uvicorn.run(
        "Orchestration.main:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
