"""
vector_store_interface.py
----------------------------
Stable contract for the vector database.

Nothing outside vector_store/ may import chromadb / langchain_chroma.
Swap engines by implementing this interface (e.g. Qdrant later).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document


class SearchResult(TypedDict):
    """Normalized hit returned to retriever / Agents."""

    id: str
    text: str
    metadata: Dict[str, Any]
    score: float  # Mümkün olduğunda [0, 1] aralığına dönüştürülmüş benzerlik skoru.


class VectorStoreInterface(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """Embed + upsert LangChain Documents. Returns assigned ids."""

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Semantic search against the collection."""

    @abstractmethod
    def delete(self, where: Dict[str, Any]) -> None:
        """Delete rows matching a metadata filter."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored chunks."""

    @abstractmethod
    def reset(self) -> None:
        """Drop and recreate the collection (full reindex)."""

    @abstractmethod
    def export_all(self) -> List[Dict[str, Any]]:
        """Export all chunks as {chunk_id, text, metadata}."""
