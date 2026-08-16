"""RecursiveCharacterTextSplitter with legal-aware separators."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from RAG.configuration.rag_config_loader import chunking_config


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunking_config.chunk_size,
        chunk_overlap=chunking_config.chunk_overlap,
        separators=list(chunking_config.separators),
        length_function=len,
    )


def split_documents(documents: List[Document]) -> List[Document]:
    if not documents:
        return []
    return get_text_splitter().split_documents(documents)
