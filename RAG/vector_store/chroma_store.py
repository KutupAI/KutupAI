"""
chroma_store.py
-----------------
LangChain + ChromaDB implementation of VectorStoreInterface.

Only this module may touch langchain_chroma / chromadb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from RAG.chroma.chroma_config import chroma_config
from RAG.embeddings.embedding_model import get_embeddings
from RAG.vector_store.vector_store_interface import SearchResult, VectorStoreInterface


class ChromaStore(VectorStoreInterface):
    def __init__(self) -> None:
        Path(chroma_config.persist_directory).mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=chroma_config.collection_name,
            embedding_function=get_embeddings(),
            persist_directory=chroma_config.persist_directory,
            collection_metadata={"hnsw:space": chroma_config.distance_metric},
        )

    @property
    def raw(self) -> Chroma:
        """Yalnız indeksleme scriptleri için gelişmiş LangChain API erişimi."""
        return self._store

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        if not documents:
            return []
        if ids is not None and len(ids) != len(documents):
            raise ValueError("ids must have the same length as documents")

        # Bu değer embedding modelinin batch boyutundan bilerek bağımsızdır.
        # HuggingFace GPU'da küçük batch kodlarken Chroma daha büyük upsert alır.
        # Kurulu Chroma sınırı okunur ve tam sınır bildiren sürümler için pay bırakılır.
        client = getattr(self._store, "_client", None)
        get_limit = getattr(client, "get_max_batch_size", None)
        chroma_limit = int(get_limit()) if callable(get_limit) else 5_000
        batch_size = max(1, min(chroma_limit, 5_000))
        assigned_ids: List[str] = []
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            batch_ids = ids[start:end] if ids is not None else None
            assigned_ids.extend(
                self._store.add_documents(
                    documents=documents[start:end],
                    ids=batch_ids,
                )
            )
        return assigned_ids

    def similarity_search(
        self,
        query: str,
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        pairs = self._store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=where,
        )

        results: List[SearchResult] = []
        for doc, distance in pairs:
            # Chroma cosine uzaklığı genelde [0, 2] aralığındadır; benzerliğe dönüştürülür.
            similarity = max(0.0, 1.0 - (float(distance) / 2.0))
            results.append(
                SearchResult(
                    id=str(doc.metadata.get("chunk_id") or doc.id or ""),
                    text=doc.page_content,
                    metadata=dict(doc.metadata or {}),
                    score=round(similarity, 4),
                )
            )
        return results

    def delete(self, where: Dict[str, Any]) -> None:
        if not where:
            return
        self._store.delete(where=where)

    def count(self) -> int:
        collection = self._store._collection  # noqa: SLF001 — count() için bilinçli iç erişim.
        return int(collection.count())

    def reset(self) -> None:
        self._store.reset_collection()

    def export_all(self) -> List[Dict[str, Any]]:
        """
        Export all stored chunks for BM25 rebuild / evaluation.
        Returns list of {chunk_id, text, metadata}.
        """
        collection = self._store._collection  # noqa: SLF001
        raw = collection.get(include=["documents", "metadatas"])
        rows: List[Dict[str, Any]] = []
        ids = raw.get("ids") or []
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        for i, doc_id in enumerate(ids):
            meta = dict(metas[i] or {})
            chunk_id = str(meta.get("chunk_id") or doc_id)
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "text": docs[i] or "",
                    "metadata": meta,
                }
            )
        return rows


_store_instance: Optional[ChromaStore] = None


def get_vector_store() -> ChromaStore:
    """Process-wide singleton used by indexer + retriever."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaStore()
    return _store_instance


def reset_vector_store_singleton() -> None:
    """Test helper — force a fresh ChromaStore on next get_vector_store()."""
    global _store_instance
    _store_instance = None
