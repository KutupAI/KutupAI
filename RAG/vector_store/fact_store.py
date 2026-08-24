"""Facts Registry için ana chunk koleksiyonundan ayrı küçük semantic indeks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from RAG.configuration.rag_config_loader import facts_registry_config, vector_store_config
from RAG.embeddings.embedding_model import get_embeddings
from RAG.vector_store.vector_store_interface import SearchResult


_TURKISH_FIELDS = {
    "decision_date": "karar tarihi hüküm günü",
    "case_number": "esas numarası dava dosya numarası",
    "decision_number": "karar numarası karar kayıt numarası",
    "effective_date": "yürürlük tarihi",
    "target_law_number": "hedef kanun numarası",
    "target_articles": "etkilenen madde maddeler",
    "amount": "süre miktarı",
    "unit": "süre birimi",
    "related_instruments": "ilgili kanun hükmünde kararname KHK",
}


def _turkish_values(values: dict[str, Any]) -> str:
    return "; ".join(
        f"{_TURKISH_FIELDS.get(key, key)}: {value}"
        for key, value in values.items()
    )


def _document(record: dict[str, Any]) -> Document:
    evidence = dict(record.get("evidence") or {})
    values = dict(record.get("values") or {})
    text = (
        f"Hukukî olgu türü: {record.get('fact_type')}. "
        f"Değerler: {json.dumps(values, ensure_ascii=False)}. "
        f"Türkçe alanlar: {_turkish_values(values)}. "
        f"Kanıt: {record.get('evidence_text', '')}"
    )
    metadata = {
        "chunk_id": f"fact:{record.get('fact_id')}",
        "fact_id": str(record.get("fact_id") or ""),
        "fact_type": str(record.get("fact_type") or "legal_fact"),
        "source_type": "legal_facts",
        "law_name": f"Yapılandırılmış hukukî olgu: {record.get('fact_type')}",
        "source_file": str(evidence.get("source_file") or "unknown"),
        "law_number": str(evidence.get("law_number") or "unknown"),
        "article_no": str(evidence.get("article_no") or "unknown"),
        "page_start": evidence.get("page_start") or 0,
        "page_end": evidence.get("page_end") or 0,
        "document_category": str(evidence.get("document_category") or "unknown"),
        "fact_values": json.dumps(values, ensure_ascii=False),
    }
    return Document(page_content=text, metadata=metadata)


class FactStore:
    def __init__(self) -> None:
        Path(vector_store_config.persist_directory).mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=facts_registry_config.collection_name,
            embedding_function=get_embeddings(),
            persist_directory=vector_store_config.persist_directory,
            collection_metadata={"hnsw:space": vector_store_config.distance_metric},
        )

    def rebuild(self, records: list[dict[str, Any]]) -> int:
        self._store.reset_collection()
        docs = [_document(record) for record in records]
        # Fact registry küçük tutulur; tek güvenli Chroma batch'i model başlatma
        # maliyetini azaltır. Ana corpus için kullanılan batch sınırı burada gerekmez.
        for start in range(0, len(docs), 5000):
            batch = docs[start : start + 5000]
            self._store.add_documents(batch, ids=[doc.metadata["chunk_id"] for doc in batch])
        return len(docs)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if not query.strip() or self.count() == 0:
            return []
        pairs = self._store.similarity_search_with_score(query, k=top_k)
        return [
            SearchResult(
                id=str(doc.metadata.get("chunk_id") or ""),
                text=doc.page_content,
                metadata=dict(doc.metadata or {}),
                score=round(max(0.0, 1.0 - float(distance) / 2.0), 4),
            )
            for doc, distance in pairs
        ]

    def count(self) -> int:
        return int(self._store._collection.count())  # noqa: SLF001


_instance: FactStore | None = None


def get_fact_store() -> FactStore:
    global _instance
    if _instance is None:
        _instance = FactStore()
    return _instance


def rebuild_fact_store(registry_path: Path) -> int:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return get_fact_store().rebuild(list(payload.get("records") or []))
