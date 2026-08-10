"""
embedding_config.py
--------------------
Settings for HuggingFace Embeddings (BAAI/bge-m3).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    embedding_dim: int = 1024
    normalize_embeddings: bool = True
    # Slightly lower default batch for CPU + large multilingual model
    batch_size: int = 16
    show_progress: bool = False


embedding_config = EmbeddingConfig()
