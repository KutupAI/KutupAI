"""
embedding_model.py
--------------------
LangChain HuggingFaceEmbeddings wrapper around BAAI/bge-m3.

This is the only place the embedding model is constructed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from RAG.embeddings.embedding_config import embedding_config


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    Singleton HuggingFaceEmbeddings (BGE-M3).

    Uses LangChain's HuggingFace integration so the same object
    plugs into LangChain Chroma / retrievers.
    """
    return HuggingFaceEmbeddings(
        model_name=embedding_config.model_name,
        model_kwargs={"device": embedding_config.device},
        encode_kwargs={
            "normalize_embeddings": embedding_config.normalize_embeddings,
            "batch_size": embedding_config.batch_size,
        },
        show_progress=embedding_config.show_progress,
    )


def embed_text(text: str) -> List[float]:
    """Embed a single query/document string."""
    return get_embeddings().embed_query(text)


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed multiple documents."""
    if not texts:
        return []
    return get_embeddings().embed_documents(texts)


def get_embedding_dim() -> int:
    return embedding_config.embedding_dim
