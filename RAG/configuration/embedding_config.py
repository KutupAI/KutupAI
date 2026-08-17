"""Backward-compatible re-export of the canonical embedding configuration."""

from RAG.embeddings.embedding_config import EmbeddingConfig, embedding_config

__all__ = ["EmbeddingConfig", "embedding_config"]
