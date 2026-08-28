"""
Application-facing Orchestration entry point.

Single path: `run_workflow` runs the full stage graph
(OCR → Classification → … → Writing). Stages that are disabled in
config.yaml are skipped (never faked). All stages are enabled by default.

Accepts either:
  * the unified Application layer envelope
    { request: {success, question, document}, ocr:{}, … writing:{} }
    (+ optional flat document_path / document_id for the temp-file hop), or
  * the legacy flat payload { document_id, document_path, question, … }.

Returns the unified layer contract: { "Success": bool, "Data": [ document, ... ] }.

Kept free of FastAPI so unit/integration tests can import without the HTTP stack.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from Orchestration.graph.graph_definition import Stage
from Orchestration.conversation_store import ConversationStore
from Orchestration.messages.message_schema import (
    contract_envelope,
    empty_document,
    is_contract_envelope,
    normalize_document,
)
from Orchestration.state.state_manager import unified_ocr_from_wire
from Orchestration.workflow.workflow_builder import AgentProtocol, build_workflow
from Orchestration.workflow.workflow_config import FallbackPolicy, WorkflowConfig, load_config

logger = logging.getLogger("Orchestration.process_service")
_conversation_store: Optional[ConversationStore] = None

_STAGE_KEYS = (
    "ocr",
    "classification",
    "extraction",
    "validation",
    "rag",
    "summary",
    "routing",
    "writing",
)


def _is_layer_envelope(payload: Dict[str, Any]) -> bool:
    request = payload.get("request")
    return isinstance(request, dict) and (
        "document" in request or "question" in request or "success" in request
    )


def _file_type_from_path(path: Path) -> str:
    return path.suffix.lstrip(".").lower()


def _build_request_section(
    *,
    document_id: str,
    document_path: Optional[str],
    question: str,
    file_name: str = "",
    file_type: str = "",
    success: bool = True,
    existing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize Application's `request` section for GraphState / Agents."""

    base = dict(existing) if isinstance(existing, dict) else {}
    document = dict(base.get("document") or {}) if isinstance(base.get("document"), dict) else {}

    document.setdefault("document_id", document_id)
    if file_name and not document.get("file_name"):
        document["file_name"] = file_name
    if file_type and not document.get("file_type"):
        document["file_type"] = file_type
    if document_path and not document.get("document_path"):
        document["document_path"] = document_path

    if not document.get("file_name") and document_path:
        document["file_name"] = Path(document_path).name
    if not document.get("file_type") and document_path:
        document["file_type"] = _file_type_from_path(Path(document_path))

    return {
        "success": bool(base.get("success", success)),
        "question": str(base.get("question") if base.get("question") is not None else question),
        "document": document,
    }


def _seed_stage_sections(state_seed: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """Ensure every pipeline stage key exists (empty object when Application sent {})."""

    for key in _STAGE_KEYS:
        value = payload.get(key)
        state_seed[key] = dict(value) if isinstance(value, dict) else {}


def _question_only_config(config: Optional[WorkflowConfig]) -> WorkflowConfig:
    """Run a question without a file through RAG → summary → writing only."""

    base = config or load_config()
    stages = dict(base.stages)
    for stage in (Stage.OCR, Stage.CLASSIFICATION, Stage.EXTRACTION, Stage.VALIDATION, Stage.ROUTING):
        stages[stage] = replace(base.stage(stage), enabled=False, fallback=FallbackPolicy.SKIP)
    return replace(base, stages=stages)


def _document_upload_config(config: Optional[WorkflowConfig]) -> WorkflowConfig:
    """Belge sorusuz yüklendiğinde kanıt araması yerine belge özeti üretir."""

    base = config or load_config()
    stages = dict(base.stages)
    for stage in (Stage.RAG, Stage.SUMMARY):
        stages[stage] = replace(base.stage(stage), enabled=False, fallback=FallbackPolicy.SKIP)
    return replace(base, stages=stages)


def run_workflow(
    *,
    document_id: str,
    document_path: Optional[str],
    accompanying_text: Optional[str] = None,
    agent_overrides: Optional[Dict[Stage, AgentProtocol]] = None,
    config: Optional[WorkflowConfig] = None,
    layer_state: Optional[Dict[str, Any]] = None,
    conversation_store: Optional[ConversationStore] = None,
) -> Dict[str, Any]:
    """Run the Orchestration graph for one request; return { Success, Data }.

    `layer_state` is the Application envelope when present; otherwise a fresh
    empty envelope is built from the flat arguments.
    """

    global _conversation_store
    question = accompanying_text or ""
    payload = dict(layer_state) if isinstance(layer_state, dict) else {}
    if conversation_store is None:
        if _conversation_store is None:
            _conversation_store = ConversationStore()
        conversation_store = _conversation_store

    # Application geçici upload klasörünü işlem sonunda siler. İlk mesajda
    # dosyayı kalıcı hafızaya kopyalamak, sonraki soruların aynı belgeyle
    # çalışmasını sağlar.
    if document_path:
        try:
            document_path = conversation_store.bind_document(document_id, document_path)
        except Exception as exc:  # Ana workflow, hafıza hatası yüzünden durmaz.
            logger.warning("conversation_document_not_saved document_id=%s error=%s", document_id, type(exc).__name__)
    else:
        document_path = conversation_store.document_path(document_id)

    # Yeni bir sohbette dosya olmadan gelen soru doğrudan RAG'a gider.
    # Aynı sohbetin daha önce kaydedilmiş belgesi varsa yukarıdaki satır
    # belge yolunu geri getirir ve normal belge akışı kullanılır.
    question_only = not document_path
    if question_only and not question:
        logger.info("process_skip document_id=%s reason=missing_document_path", document_id)
        return contract_envelope(
            False,
            [empty_document(document_id=document_id, question=question)],
        )
    if question_only:
        try:
            conversation_store.ensure_conversation(document_id)
        except Exception as exc:
            logger.warning("conversation_not_initialized document_id=%s error=%s", document_id, type(exc).__name__)

    path = Path(document_path) if document_path else None
    document_upload_only = path is not None and not question
    if path is not None and not path.is_file():
        logger.info("process_skip document_id=%s reason=path_missing", document_id)
        return contract_envelope(
            False,
            [
                empty_document(
                    document_id=document_id,
                    file_name=path.name,
                    file_type=path.suffix.lstrip("."),
                    question=question,
                )
            ],
        )

    existing_request = payload.get("request") if _is_layer_envelope(payload) else None
    request_section = _build_request_section(
        document_id=document_id,
        document_path=str(path) if path is not None else None,
        question=question,
        file_name=path.name if path is not None else "",
        file_type=_file_type_from_path(path) if path is not None else "",
        success=True,
        existing=existing_request if isinstance(existing_request, dict) else None,
    )
    memory_context = conversation_store.search(document_id, question)
    document_hash = conversation_store.document_hash(document_id) if path is not None else ""
    cached_ocr = None
    if path is not None:
        try:
            cached_ocr = conversation_store.get_ocr_cache(document_hash)
        except Exception as exc:  # OCR cache ana workflow'u engellemez.
            logger.warning("ocr_cache_read_failed document_id=%s error=%s", document_id, type(exc).__name__)

    # Flat fields Agents / StateManager still lift to the top level.
    workflow_request: Dict[str, Any] = {
        "document_id": document_id,
        "document_path": str(path) if path is not None else "",
        "accompanying_text": question or None,
        "question": question or None,
        "text": question or None,
        "success": request_section["success"],
        "document": request_section["document"],
        # Nested Application contract (Agents read state["request"]).
        "request": request_section,
        # Bu alanlar sadece Orchestration içindedir; dış katman sözleşmesine
        # eklenmez. Zayıf eşleşmede boş kalır ve geçmiş kullanılmaz.
        "conversation_memory": memory_context.for_writer() if memory_context.is_follow_up else "",
        "conversation_focus_law": memory_context.focus_law if memory_context.is_follow_up else "",
        "conversation_reference_law": memory_context.reference_law if memory_context.is_follow_up else "",
        "conversation_is_follow_up": memory_context.is_follow_up,
        "document_upload_only": document_upload_only,
        "writer_instruction": (
            "Yüklenen belgeyi Türkçe olarak kısa ve anlaşılır biçimde özetle. "
            "Belgenin türünü, ana konusunu ve belgede açıkça yer alan önemli bilgileri belirt. "
            "Belgede olmayan bilgileri ekleme."
            if document_upload_only
            else ""
        ),
    }
    _seed_stage_sections(workflow_request, payload)
    if cached_ocr:
        # Bu işaret yalnızca Orchestration içindir. Workflow OCR ajanını
        # çağırmadan hazır sonucu uygular; dış sözleşmeye eklenmez.
        workflow_request["ocr"] = cached_ocr
        workflow_request["ocr_cache_hit"] = True

    logger.info(
        "workflow_start document_id=%s path=%s has_text=%s envelope=%s ocr_cache=%s",
        document_id,
        path.name if path is not None else "question_only",
        bool(question),
        _is_layer_envelope(payload),
        "hit" if cached_ocr else "miss",
    )

    active_config = (
        _question_only_config(config)
        if question_only
        else _document_upload_config(config)
        if document_upload_only
        else config
    )
    workflow = build_workflow(
        config=active_config,
        agent_overrides=agent_overrides,
    )
    result = workflow.run(workflow_request)
    state = result.state

    ocr_payload = state.get("ocr_result")
    if is_contract_envelope(ocr_payload):
        data = ocr_payload.get("Data") or []
        docs = [normalize_document(item) for item in data if isinstance(item, dict)]
        doc = docs[0] if docs else empty_document(document_id=document_id, question=question)
    else:
        doc = empty_document(document_id=document_id, question=question)

    if not doc.get("document_id"):
        doc["document_id"] = document_id
    if question and not doc.get("question"):
        doc["question"] = question
    if path is not None and not doc.get("file_name"):
        doc["file_name"] = path.name
    if path is not None and not doc.get("file_type"):
        doc["file_type"] = path.suffix.lstrip(".")

    # Prefer unified short keys; fall back to Orchestration wire mirrors.
    for extra_key, state_key in (
        ("classification", "classification_result"),
        ("extraction", "extraction_result"),
        ("validation", "validation_result"),
        ("rag", "rag_result"),
        ("summary", "summary"),
        ("routing", "routing"),
        ("writing", "writing"),
    ):
        value = state.get(extra_key) or state.get(state_key)
        # Legacy mirror while older mocks still wrote routing_decision.
        if extra_key == "routing" and not value:
            value = state.get("routing_decision")
        # Prefer unified rag {success, rag_data} over wire {success, data}.
        if extra_key == "rag":
            short = state.get("rag")
            if isinstance(short, dict) and short:
                value = short
            elif isinstance(value, dict) and "rag_data" not in value and "data" in value:
                value = {
                    "success": bool(value.get("success")),
                    "rag_data": value.get("data")
                    if isinstance(value.get("data"), dict)
                    else {
                        "operation": "retrieve",
                        "query": "",
                        "results": value.get("data") if isinstance(value.get("data"), list) else [],
                    },
                }
            # Always emit a rag object (never omit / leave seeded {}).
            if not value:
                value = {
                    "success": False,
                    "rag_data": {"operation": "retrieve", "query": "", "results": []},
                    "error": {
                        "code": "rag_missing",
                        "message": "RAG stage did not produce a payload.",
                    },
                }
        if value:
            doc[extra_key] = value

    # Final safety: rag key must exist on the document returned to Application.
    if not isinstance(doc.get("rag"), dict) or not doc.get("rag"):
        doc["rag"] = {
            "success": False,
            "rag_data": {"operation": "retrieve", "query": "", "results": []},
            "error": {
                "code": "rag_missing",
                "message": "RAG stage did not produce a payload.",
            },
        }

    # Always attach unified `ocr` for Application / Presentation.
    if isinstance(state.get("ocr"), dict) and state["ocr"]:
        doc["ocr"] = state["ocr"]
    elif is_contract_envelope(ocr_payload):
        doc["ocr"] = unified_ocr_from_wire(ocr_payload)

    if path is not None and not cached_ocr:
        try:
            conversation_store.save_ocr_cache(document_hash, state.get("ocr"))
        except Exception as exc:  # Cache yazılamasa da yanıt kullanıcıya döner.
            logger.warning("ocr_cache_write_failed document_id=%s error=%s", document_id, type(exc).__name__)

    success = result.completed and not result.terminated
    # Treat a successful OCR envelope as overall success when later stages
    # were skipped (not failed).
    if not success and is_contract_envelope(ocr_payload) and ocr_payload.get("Success"):
        success = True

    writing = doc.get("writing") if isinstance(doc.get("writing"), dict) else {}
    answer = str(writing.get("answer") or "")
    history_question = question or (f"Belge yüklendi: {path.name}" if document_upload_only and path else "")
    try:
        conversation_store.record_turn(
            document_id,
            history_question,
            answer,
            doc.get("rag"),
            # Sidebar'da Detay/Kaynak açılması için yalnız gerekli UI verisini sakla.
            # OCR metni burada tutulmaz; büyük belge metni hafızayı şişirmemelidir.
            {
                key: doc.get(key, {})
                for key in ("classification", "extraction", "validation", "rag", "summary", "routing", "writing")
            },
        )
    except Exception as exc:
        logger.warning("conversation_turn_not_saved document_id=%s error=%s", document_id, type(exc).__name__)

    logger.info(
        "workflow_done workflow_id=%s document_id=%s success=%s stages_run=%s",
        state.get("workflow_id"),
        document_id,
        success,
        len(state.get("history", [])),
    )
    return contract_envelope(success, [doc])


def run_workflow_from_application(payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """Entry used by FastAPI: accept Application envelope or legacy flat body."""

    payload = dict(payload or {})
    question = ""
    for key in ("accompanying_text", "text", "question"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            question = value.strip()
            break

    document_id = str(payload.get("document_id") or "")
    document_path = payload.get("document_path")
    layer_state: Optional[Dict[str, Any]] = None

    if _is_layer_envelope(payload):
        layer_state = payload
        request = payload.get("request") or {}
        document = request.get("document") if isinstance(request, dict) else {}
        if isinstance(document, dict):
            if not document_id:
                document_id = str(document.get("document_id") or "")
            if not document_path:
                document_path = document.get("document_path")
            if not question and isinstance(request.get("question"), str):
                question = request["question"]

    if not document_id:
        document_id = "unknown"

    started = time.monotonic()
    response = run_workflow(
        document_id=document_id,
        document_path=str(document_path) if document_path else None,
        accompanying_text=question or None,
        layer_state=layer_state,
        **kwargs,
    )
    logger.info(
        "request_done document_id=%s total=%.2fs",
        document_id,
        time.monotonic() - started,
    )
    return response


# Backward-compatible name used by older imports/docs.
run_full_workflow = run_workflow
