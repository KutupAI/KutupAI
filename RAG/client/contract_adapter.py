"""RAG katmanı için ekipler arası JSON sözleşmesi adaptörü.

Bu modül HTTP sunucusu değildir. Application katmanı gelen JSON nesnesini
``handle_rag_request`` fonksiyonuna verir; RAG de yalnız sözleşmede tanımlı
``success``, ``data`` ve ``error`` alanlarıyla yanıt döndürür.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from langchain_core.documents import Document

from RAG.agent.legal_agent import LegalRagAgent
from RAG.ingestion.pipeline import IngestionReport, ingest_contract_document
from RAG.retriever.query_router import choose_query_plan
from RAG.retriever.retriever import retrieve


ContractPayload = Mapping[str, Any]
_HUKUKI_ATIF = re.compile(
    r"\b(?:\d{3,5}\s*(?:sayılı|sayili)|KHK\s*[-/]?\s*\d+|\d+\s*(?:inci|ıncı|uncu|üncü)\s*madde)\b",
    re.IGNORECASE,
)
_KANUN_ATFI = re.compile(r"\b(\d{3,5})\s*(?:sayılı|sayili)\b", re.IGNORECASE)
_HEDEF_KANUN_BAGLAMI = re.compile(
    r"\b(\d{3,5})\s*(?:sayılı|sayili)\b.{0,120}?\bkanun(?:un|u|ın|in)?\s+"
    r"(?:bakımından|icin|için|kapsamında|kapsaminda|degisiklik\s+cetvelinde|değişiklik\s+cetvelinde)\b",
    re.IGNORECASE | re.DOTALL,
)
_DEGISIKLIK_SORUSU = re.compile(
    r"\b(?:değişiklik|degisiklik|değiştiren|degistiren|etkil(?:eyen|ediği|edigi)|"
    r"yürürlüğe\s+giriş|yururluge\s+giris|düzenlem\w*|duzenlem\w*|cetvel)\b",
    re.IGNORECASE,
)

# Sık kullanılan kısaltmalar soru içinde kanun numarası yazılmadan da geçer.
# Bu eşleştirme yalnız retrieval sorgusunu daraltır; dış sözleşmeye alan eklemez.
_KANUN_KISALTMALARI = {
    "kvkk": "6698",
}


def _error(code: str, message: str, *, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Sözleşmenin her hata durumunda aynı üst seviye şekli korur."""
    return {
        "success": False,
        "data": None,
        "error": {"code": code, "message": message, "details": details or {}},
    }


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _require_data(payload: ContractPayload) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("İstek JSON nesnesi olmalıdır.")
    if payload.get("success") is False:
        raise ValueError("Başarısız bir önceki katman çıktısı RAG'a indekslenemez.")
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("İstek içinde 'data' nesnesi bulunmalıdır.")
    return dict(data)


def _pages_to_text(pages: Any) -> str:
    """Sayfa dizisini chunker'ın kullandığı güvenli sayfa işaretlerine çevirir."""
    if not isinstance(pages, list):
        return ""
    rows: List[tuple[int, str]] = []
    for index, page in enumerate(pages, start=1):
        if not isinstance(page, Mapping):
            continue
        text = str(page.get("text") or "").strip()
        if not text:
            continue
        number = page.get("page_number", index)
        try:
            page_number = max(1, int(number))
        except (TypeError, ValueError):
            page_number = index
        rows.append((page_number, text))
    rows.sort(key=lambda item: item[0])
    return "\n\n".join(f"[[RAG_PAGE:{number}]]\n{text}" for number, text in rows)


def _document_from_index_data(data: Dict[str, Any]) -> Document:
    """INDEX sözleşmesini dosya sistemi gerektirmeyen LangChain Document'e dönüştürür."""
    document_id = _clean_text(data.get("document_id"))
    file_name = Path(_clean_text(data.get("file_name"))).name
    if not document_id:
        raise ValueError("INDEX isteğinde 'document_id' zorunludur.")
    if not file_name:
        raise ValueError("INDEX isteğinde 'file_name' zorunludur.")

    # Sayfalar verilmişse sayfa numarası kaynak doğruluğu için önceliklidir.
    content = _pages_to_text(data.get("pages")) or str(data.get("full_text") or "").strip()
    if not content:
        raise ValueError("INDEX isteğinde 'pages' veya 'full_text' içinde metin bulunmalıdır.")

    supplied_metadata = data.get("metadata")
    metadata = dict(supplied_metadata) if isinstance(supplied_metadata, Mapping) else {}
    language = data.get("language")
    if isinstance(language, Mapping) and language.get("detected"):
        metadata["language"] = str(language["detected"])
    metadata.update(
        {
            "document_id": document_id,
            "source": file_name,
            "source_file": file_name,
            "source_type": str(metadata.get("source_type") or "uploads"),
            "source_dir": str(metadata.get("source_dir") or "contract"),
            "law_name": str(metadata.get("law_name") or Path(file_name).stem),
        }
    )
    return Document(page_content=content, metadata=metadata)


def _chunk_payload(chunk: Document) -> Dict[str, Any]:
    """Chroma'ya yazılmış chunk'ı INDEX çıktı şemasına indirger."""
    metadata = dict(chunk.metadata or {})
    return {
        "chunk_id": str(metadata.get("chunk_id") or ""),
        "text": chunk.page_content,
        "law_number": str(metadata.get("law_number") or "unknown"),
        "article_no": str(metadata.get("article_no") or metadata.get("article_number") or "unknown"),
        "article_type": str(metadata.get("article_type") or "unknown"),
        "page_start": metadata.get("page_start") or metadata.get("page"),
        "page_end": metadata.get("page_end") or metadata.get("page"),
    }


def _index(data: Dict[str, Any]) -> Dict[str, Any]:
    document = _document_from_index_data(data)
    report, chunks = ingest_contract_document(document)
    metadata = dict(document.metadata or {})
    return {
        "success": True,
        "data": {
            "operation": "index",
            "document_id": metadata["document_id"],
            "file_name": metadata["source_file"],
            "law_number": str(chunks[0].metadata.get("law_number") if chunks else "unknown"),
            "chunks_created": len(chunks),
            "indexed": True,
            "vector_store": {"backend": "ChromaDB", "collection": "legal_documents"},
            "metrics": {
                "vectors_total": report.vector_count,
                "invalid_metadata": report.invalid_metadata,
            },
            "chunks": [_chunk_payload(chunk) for chunk in chunks],
        },
        "error": None,
    }


def _memory_text(memory: Any) -> Optional[str]:
    """Sözleşmedeki role/content geçmişini Agent'ın sınırlı bellek metnine dönüştürür."""
    if not isinstance(memory, list):
        return None
    rows: List[str] = []
    for item in memory[-3:]:
        if not isinstance(item, Mapping):
            continue
        role = _clean_text(item.get("role")) or "unknown"
        content = _clean_text(item.get("content"))
        if content:
            rows.append(f"{role}: {content[:700]}")
    return "\n".join(rows) or None


def _query_text(question: str, filters: Any) -> str:
    """Haricî metadata filtrelerini mevcut router'ın anlayacağı açık atfa dönüştürür."""
    if not isinstance(filters, Mapping):
        return question
    law_number = _clean_text(filters.get("law_number"))
    article_no = _clean_text(filters.get("article_no") or filters.get("article_number"))
    if law_number and article_no:
        return f"{law_number} sayılı Kanunun {article_no}. maddesi kapsamında: {question}"
    if law_number:
        return f"{law_number} sayılı Kanun kapsamında: {question}"
    return question


def _public_source(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Agent kaynak kaydını QUERY sözleşmesindeki alan isimlerine çevirir."""
    return {
        "label": str(source.get("label") or ""),
        "chunk_id": str(source.get("chunk_id") or ""),
        "document_id": str(source.get("document_id") or ""),
        "law_number": str(source.get("law_number") or "unknown"),
        "law_name": str(source.get("law_name") or "Bilinmeyen kanun"),
        "article_no": str(source.get("article_number") or source.get("article_no") or "unknown"),
        "page_start": source.get("page_start"),
        "page_end": source.get("page_end"),
        "text": str(source.get("text") or ""),
        "score": float(source.get("score") or 0.0),
    }


def _contract_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Retriever sonucunu yeni katman sözleşmesinin sade kaynak biçimine çevirir."""
    metadata = dict(result.get("metadata") or {})
    return {
        "chunk_id": str(metadata.get("chunk_id") or result.get("id") or ""),
        "law_number": str(metadata.get("law_number") or "unknown"),
        "law_name": str(metadata.get("law_name") or metadata.get("source_file") or "Bilinmeyen kaynak"),
        "article_no": str(metadata.get("article_no") or metadata.get("article_number") or "unknown"),
        "page_start": metadata.get("page_start") or metadata.get("page"),
        "page_end": metadata.get("page_end") or metadata.get("page"),
        "text": str(result.get("text") or ""),
        "score": float(result.get("score") or 0.0),
    }


def _ocr_hedef_kanun(question: str, full_text: str) -> str | None:
    """OCR'deki tek açık hedef kanunu kısa sorgu bağlamına taşır."""
    if not question or not full_text or not _DEGISIKLIK_SORUSU.search(question):
        return None
    question_laws = {match.group(1) for match in _KANUN_ATFI.finditer(question)}
    candidates = {
        match.group(1)
        for match in _HEDEF_KANUN_BAGLAMI.finditer(full_text)
        if match.group(1) not in question_laws
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _soru_kanun_kisaltmasi(question: str) -> str | None:
    """Sorudaki bilinen kanun kısaltmasını kanun numarasına çevirir."""
    normalized = question.casefold()
    for abbreviation, law_number in _KANUN_KISALTMALARI.items():
        if re.search(rf"\b{re.escape(abbreviation)}\b", normalized):
            return law_number
    return None


def _state_query(state: Mapping[str, Any]) -> str:
    """State içindeki soru ve belge sinyallerinden LLM'siz retrieval sorgusu kurar."""
    request = state.get("request")
    question = _clean_text(request.get("question")) if isinstance(request, Mapping) else ""
    # Değişiklik sorusunda OCR'den yalnız açık hedef kanunu al.
    abbreviated_law = _soru_kanun_kisaltmasi(question) if question else None
    is_legal_question = bool(question and (_HUKUKI_ATIF.search(question) or abbreviated_law))

    classification = state.get("classification")
    document_type = ""
    if isinstance(classification, Mapping) and classification.get("success"):
        document_type = _clean_text(classification.get("document_type"))
    ocr = state.get("ocr")
    ocr_full_text = ""
    full_text = ""
    if isinstance(ocr, Mapping) and ocr.get("success"):
        ocr_data = ocr.get("ocr_data")
        if isinstance(ocr_data, Mapping):
            # Belge metninin tümünü sorguya koymak yerine sınırlı bir kesit kullanılır;
            # bu, OCR içeriğinin retrieval'a katkı sağlamasını ve gecikmenin sabit kalmasını sağlar.
            # Tüm metin hedef kanunu bulmak için okunur, sorguya eklenmez.
            ocr_full_text = _clean_text(ocr_data.get("full_text"))
            full_text = ocr_full_text[:1200]
    if is_legal_question:
        target_law = _ocr_hedef_kanun(question, ocr_full_text) or abbreviated_law
        parts = [question]
        if target_law:
            parts.append(f"Hedef kanun: {target_law} sayılı Kanun")
        # Takip sorusundaki “bu düzenleme” için önceki değişikliği ekleriz.
        # Açıkça sorulan kanun (ör. 7196) karşılaştırma tabanı olarak kalır.
        reference_law = _clean_text(state.get("conversation_reference_law"))
        if bool(state.get("conversation_is_follow_up")) and reference_law:
            parts.append(f"Önceki düzenleme: {reference_law} sayılı Kanun")
        return "\n".join(parts)
    parts = [question]
    # Sadece hafıza katmanı güçlü bir takip ilişkisi kurduysa önceki kanun
    # odağını retrieval sorgusuna ekleriz. Açık yeni kanun atfı her zaman
    # önceliklidir; böylece eski konu yeni soruyu daraltmaz.
    focus_law = _clean_text(state.get("conversation_focus_law"))
    is_follow_up = bool(state.get("conversation_is_follow_up"))
    if is_follow_up and focus_law and not _KANUN_ATFI.search(question):
        parts.append(f"Önceki konuşma odağı: {focus_law} sayılı Kanun")
    if document_type:
        parts.append(f"Belge türü: {document_type}")
    if full_text:
        parts.append(f"Belge metni: {full_text}")
    return "\n".join(part for part in parts if part)


def handle_layer_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Yeni Layers_contracts state'ini LLM çağırmadan işler.

    Girdi state'i korunur; yalnız ``rag`` alanı doldurulur. Sonuçta answer,
    context_for_llm veya LLM çağrısı yoktur. RAG'in sorumluluğu yalnız ilgili
    hukukî pasajları sıralı olarak bulup sonraki katmana aktarmaktır.
    """
    output = deepcopy(dict(state))
    request = output.get("request")
    if not isinstance(request, Mapping) or not request.get("success"):
        output["rag"] = {
            "success": False,
            "rag_data": {"operation": "retrieve", "query": "", "results": []},
            "error": {"code": "invalid_request", "message": "request.success ve request.question zorunludur."},
        }
        return output

    query = _state_query(output)
    if not query:
        output["rag"] = {
            "success": False,
            "rag_data": {"operation": "retrieve", "query": "", "results": []},
            "error": {"code": "empty_question", "message": "request.question boş olamaz."},
        }
        return output

    try:
        plan = choose_query_plan(query)
        results = retrieve(
            query,
            top_k=5,
            mode=plan.mode,
            use_prf=plan.use_prf,
            use_reranker=plan.use_reranker,
            use_graph=plan.use_graph,
            # Bu katman sözleşmesinde hiçbir LLM kullanılmaz. Sadece hızlı,
            # deterministik yazım düzeltmesi ve retrieval modelleri çalışır.
            use_query_transform_llm=False,
        )
        # Reranker adayların tamamını eleyebilir. Bu durumda aynı sorguyu
        # ilk retrieval sırasıyla tekrar denemek, mevcut kanıtı kaybetmez.
        if not results and plan.use_reranker:
            results = retrieve(
                query,
                top_k=5,
                mode=plan.mode,
                use_prf=plan.use_prf,
                use_reranker=False,
                use_graph=plan.use_graph,
                use_query_transform_llm=False,
            )
        output["rag"] = {
            "success": True,
            "rag_data": {
                "operation": "retrieve",
                "query": query,
                "results": [_contract_result(result) for result in results],
            },
        }
    except Exception as exc:
        output["rag"] = {
            "success": False,
            "rag_data": {"operation": "retrieve", "query": query, "results": []},
            "error": {"code": "retrieval_failed", "message": "RAG retrieval tamamlanamadı.", "type": type(exc).__name__},
        }
    return output


def _query(data: Dict[str, Any], agent_factory: Callable[[], LegalRagAgent]) -> Dict[str, Any]:
    question = _clean_text(data.get("question"))
    if not question:
        raise ValueError("QUERY isteğinde 'question' zorunludur.")
    try:
        top_k = max(1, min(20, int(data.get("top_k", 5))))
    except (TypeError, ValueError):
        top_k = 5

    started = perf_counter()
    agent = agent_factory()
    answer = agent.answer(
        _query_text(question, data.get("filters")),
        top_k=top_k,
        conversation_memory=_memory_text(data.get("conversation_memory")),
    )
    sources = [_public_source(source) for source in answer.sources]
    context_rows = [
        f"[{source['label']}] {source['law_name']} | Kanun {source['law_number']} | "
        f"Madde {source['article_no']} | Sayfa {source['page_start']}\n{source['text']}"
        for source in sources
    ]
    return {
        "success": True,
        "data": {
            "operation": "query",
            "session_id": _clean_text(data.get("session_id")) or None,
            "question": question,
            "retrieval_plan": answer.retrieval_plan,
            "grounded": answer.grounded,
            "sources": sources,
            "context_for_llm": "\n\n".join(context_rows),
            "answer": answer.answer,
            "metrics": {
                "retrieval_ms": answer.retrieval_ms,
                "generation_ms": answer.generation_ms,
                "total_ms": answer.total_ms or round((perf_counter() - started) * 1000, 3),
                "cache_hit": answer.cache_hit,
            },
            "refusal_reason": answer.refusal_reason,
        },
        "error": None,
    }


def handle_rag_request(
    payload: ContractPayload,
    *,
    agent_factory: Callable[[], LegalRagAgent] = LegalRagAgent,
) -> Dict[str, Any]:
    """Eski JSON sözleşmesini işler; state sözleşmesini otomatik yönlendirir."""
    # Yeni Layers_contracts formatı üst seviyede request/ocr/classification
    # alanlarını taşır ve kesinlikle LLM çağırmaz.
    if isinstance(payload, Mapping) and "request" in payload:
        return handle_layer_state(payload)
    try:
        data = _require_data(payload)
        operation = _clean_text(data.get("operation")).casefold()
        if operation == "index":
            return _index(data)
        if operation == "query":
            return _query(data, agent_factory)
        return _error("unsupported_operation", "operation yalnızca 'index' veya 'query' olabilir.")
    except ValueError as exc:
        return _error("validation_error", str(exc))
    except Exception as exc:  # Katman sınırında ham traceback dışarı sızdırılmaz.
        return _error("rag_processing_error", "RAG isteği işlenemedi.", details={"type": type(exc).__name__})
