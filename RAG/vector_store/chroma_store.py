"""
chroma_store.py
------------------
التطبيق الفعلي الحالي لـ VectorStoreInterface باستخدام ChromaDB.

هذا هو الملف الوحيد المسموح له باستيراد مكتبة chromadb مباشرة.
"""

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from RAG.configuration.chroma_config import chroma_config
from RAG.vector_store.vector_store_interface import SearchResult, VectorStoreInterface


class ChromaStore(VectorStoreInterface):
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=chroma_config.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=chroma_config.collection_name,
            metadata={"hnsw:space": chroma_config.distance_metric},
        )

    def add_documents(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        results: List[SearchResult] = []
        if not raw["ids"] or not raw["ids"][0]:
            return results

        ids = raw["ids"][0]
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for i in range(len(ids)):
            # ChromaDB يرجع "distance" (كلما قل كان أقرب) - نحوّلها لـ score تشابه
            # حيث كلما اقترب من 1 كان التشابه أعلى (يفترض مسافة cosine بمدى [0, 2])
            similarity_score = 1.0 - (distances[i] / 2.0)
            results.append(
                SearchResult(
                    id=ids[i],
                    text=docs[i],
                    metadata=metas[i] or {},
                    score=round(similarity_score, 4),
                )
            )
        return results

    def delete(self, where: Dict[str, Any]) -> None:
        self._collection.delete(where=where)

    def count(self) -> int:
        return self._collection.count()


# نسخة وحيدة (singleton) يُعاد استخدامها في كل الطبقة
_store_instance: Optional[ChromaStore] = None


def get_vector_store() -> ChromaStore:
    """نقطة الوصول الموحّدة للحصول على نسخة من مخزن المتجهات."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaStore()
    return _store_instance
