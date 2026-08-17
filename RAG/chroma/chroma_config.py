"""
chroma_config.py
------------------
ChromaDB kalıcılık ayarları. Değerler merkezi YAML ayarından gelir.
"""

from __future__ import annotations

from dataclasses import dataclass
from RAG.configuration.rag_config_loader import vector_store_config


@dataclass(frozen=True)
class ChromaConfig:
    persist_directory: str = str(vector_store_config.persist_directory)
    collection_name: str = vector_store_config.collection_name
    # Cosine metriği normalize edilmiş BGE-M3 vektörleriyle uyumludur.
    distance_metric: str = vector_store_config.distance_metric


chroma_config = ChromaConfig()
