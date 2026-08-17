"""
embedding_model.py
--------------------
LangChain HuggingFaceEmbeddings wrapper around BAAI/bge-m3.
KutupAI Optimized: GPU acceleration ve hata toleransı eklendi.
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
    CUDA desteği ile hızlandırılmıştır.
    """
    try:
        return HuggingFaceEmbeddings(
            model_name=embedding_config.model_name,
            model_kwargs={"device": embedding_config.device},
            encode_kwargs={
                "normalize_embeddings": embedding_config.normalize_embeddings,
                "batch_size": embedding_config.batch_size,
            },
            show_progress=embedding_config.show_progress,
        )
    except Exception as e:
        print(f" Embedding modeli yüklenirken hata oluştu: {e}")
        print(" Lütfen internet bağlantınızı kontrol edin veya modelin önbelleğe alındığından emin olun.")
        raise e


def embed_text(text: str) -> List[float]:
    """Tek bir sorgu/metin string'ini vektöre dönüştürür."""
    return get_embeddings().embed_query(text)


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Birden fazla belgeyi toplu halde vektöre dönüştürür."""
    if not texts:
        return []
    return get_embeddings().embed_documents(texts)


def get_embedding_dim() -> int:
    return embedding_config.embedding_dim