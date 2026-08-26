"""
KutupAI - Belge Zenginleştirici (Document Enricher)
---------------------------------------------------
Chunk'ları ChromaDB ve BM25'e yüklemeden önce kararlı (stable) chunk_id 
ve atıf (citation) alanlarıyla zenginleştirir. KutupAI chunker.py ile tam uyumludur.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from RAG.metadata.schema import ChunkMetadata

# Madde tespiti için yedek düzenli ifade.
_ARTICLE_RE = re.compile(r"(?:MADDE|Madde)\s+(\d+)", re.UNICODE)


def _article(text: str) -> str:
    match = _ARTICLE_RE.search(text)
    return match.group(1) if match else "unknown"


def _chunk_id(source_file: str, index: int, text: str) -> str:
    """Dosya adı, indeks ve metnin ilk 64 karakterine göre benzersiz hash üretir."""
    raw = f"{source_file}:{index}:{text[:64]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def enrich_documents(chunks: List[Document]) -> List[Document]:
    """
    Chunk'ları ChromaDB şemasına uygun hale getirir.
    🚀 KutupAI chunker.py'den gelen 'article_no' ve 'law_number' alanlarını korur.
    """
    enriched: List[Document] = []
    for index, chunk in enumerate(chunks):
        meta = dict(chunk.metadata or {})
        source_file = str(meta.get("source_file") or Path(str(meta.get("source", "unknown"))).name)
        page = meta.get("page_start", meta.get("page"))
        page_int = int(page) if isinstance(page, (int, float)) or (
            isinstance(page, str) and str(page).isdigit()
        ) else None

        # Chunker verisi yoksa madde numarasını metinden çıkar.
        kutupai_article_no = meta.get("article_no")
        final_article_number = str(kutupai_article_no) if kutupai_article_no is not None else str(meta.get("article_number") or _article(chunk.page_content))
        
        final_law_number = str(meta.get("law_number", "unknown"))

        chunk_meta = ChunkMetadata(
            chunk_id=meta.get("chunk_id") or _chunk_id(source_file, index, chunk.page_content),
            source_file=source_file,
            source_type=str(meta.get("source_type", "unknown")),
            law_name=str(meta.get("law_name") or Path(source_file).stem),
            article_number=final_article_number, # ChromaDB şeması article_number bekliyor
            chunk_index=int(meta.get("chunk_index", index)),
            law_number=final_law_number, # Hibrit filtreleme için kritik
            effective_date=str(meta.get("effective_date", "unknown")),
            language=str(meta.get("language", "tr")),
            page=page_int,
        )
        
        clean = chunk_meta.to_chroma_dict()
        
        # Filtreleme alanlarını Chroma metadatasında koru.
        kutupai_specific_keys = ["article_no", "article_type", "content_type", "chunk_length"]
        for key in kutupai_specific_keys:
            if key in meta and meta[key] is not None:
                clean[key] = str(meta[key]) if not isinstance(meta[key], (int, float, bool)) else meta[key]

        for key, value in meta.items():
            if key not in clean and isinstance(value, (str, int, float, bool)):
                clean[key] = value
                
        enriched.append(Document(page_content=chunk.page_content, metadata=clean))
        
    return enriched
