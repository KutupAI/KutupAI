"""
chroma_config.py
------------------
ChromaDB persistence settings. Imported only by vector_store/chroma_store.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_RAG_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ChromaConfig:
    persist_directory: str = str((_RAG_ROOT / "documents" / ".chroma_db").resolve())
    collection_name: str = "legal_documents"
    # Cosine distance — matches normalized BGE-M3 vectors
    distance_metric: str = "cosine"


chroma_config = ChromaConfig()
