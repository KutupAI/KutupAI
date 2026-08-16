"""Attach stable chunk_id + citation fields before upsert."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from RAG.metadata.schema import ChunkMetadata

_ARTICLE_RE = re.compile(r"(?:MADDE|Madde)\s+(\d+)", re.UNICODE)


def _article(text: str) -> str:
    match = _ARTICLE_RE.search(text)
    return match.group(1) if match else "unknown"


def _chunk_id(source_file: str, index: int, text: str) -> str:
    raw = f"{source_file}:{index}:{text[:64]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def enrich_documents(chunks: List[Document]) -> List[Document]:
    enriched: List[Document] = []
    for index, chunk in enumerate(chunks):
        meta = dict(chunk.metadata or {})
        source_file = str(meta.get("source_file") or Path(str(meta.get("source", "unknown"))).name)
        page = meta.get("page")
        page_int = int(page) if isinstance(page, (int, float)) or (
            isinstance(page, str) and str(page).isdigit()
        ) else None

        chunk_meta = ChunkMetadata(
            chunk_id=_chunk_id(source_file, index, chunk.page_content),
            source_file=source_file,
            source_type=str(meta.get("source_type", "unknown")),
            law_name=str(meta.get("law_name") or Path(source_file).stem),
            article_number=str(meta.get("article_number") or _article(chunk.page_content)),
            chunk_index=index,
            law_number=str(meta.get("law_number", "unknown")),
            effective_date=str(meta.get("effective_date", "unknown")),
            language=str(meta.get("language", "tr")),
            page=page_int,
        )
        clean = chunk_meta.to_chroma_dict()
        for key, value in meta.items():
            if key not in clean and isinstance(value, (str, int, float, bool)):
                clean[key] = value
        enriched.append(Document(page_content=chunk.page_content, metadata=clean))
    return enriched
