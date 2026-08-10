"""Ingestion: upload → load → chunk → metadata → Chroma + BM25."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from RAG.configuration.rag_config_loader import documents_config
from RAG.ingestion.chunker import split_documents
from RAG.ingestion.enricher import enrich_documents
from RAG.ingestion.loader import load_all_sources, load_directory
from RAG.ingestion.uploader import SourceBucket, upload_file
from RAG.metadata.registry import apply_source_metadata
from RAG.metadata.schema import SourceMetadata, validate_chunk_metadata
from RAG.retriever.bm25_index import rebuild_bm25_from_chunks, reset_bm25_singleton
from RAG.vector_store.chroma_store import get_vector_store, reset_vector_store_singleton


@dataclass
class IngestionReport:
    files_indexed: Dict[str, int]
    total_chunks: int
    vector_count: int
    invalid_metadata: int = 0


def _prepare(documents: List[Document]) -> List[Document]:
    return enrich_documents(split_documents(apply_source_metadata(documents)))


def _upsert(chunks: List[Document]) -> int:
    if not chunks:
        return 0
    invalid = sum(1 for c in chunks if validate_chunk_metadata(c.metadata))
    ids = [str(c.metadata["chunk_id"]) for c in chunks]
    get_vector_store().add_documents(documents=chunks, ids=ids)
    return invalid


def ingest_documents(documents: List[Document], *, rebuild_bm25: bool = True) -> IngestionReport:
    by_file: Dict[str, List[Document]] = {}
    for doc in documents:
        by_file.setdefault(str(doc.metadata.get("source_file", "unknown")), []).append(doc)

    summary: Dict[str, int] = {}
    invalid = 0
    for name, docs in by_file.items():
        chunks = _prepare(docs)
        invalid += _upsert(chunks)
        summary[name] = len(chunks)

    if rebuild_bm25:
        rebuild_bm25_from_chunks(get_vector_store().export_all())

    return IngestionReport(
        files_indexed=summary,
        total_chunks=sum(summary.values()),
        vector_count=get_vector_store().count(),
        invalid_metadata=invalid,
    )


def ingest_directory(directory: Path, *, rebuild_bm25: bool = True) -> IngestionReport:
    return ingest_documents(load_directory(directory), rebuild_bm25=rebuild_bm25)


def ingest_file(
    file_path: Path | str,
    *,
    bucket: SourceBucket = "uploads",
    metadata: Optional[SourceMetadata] = None,
    copy_into_bucket: bool = True,
) -> IngestionReport:
    path = Path(file_path)
    if copy_into_bucket:
        path = upload_file(path, bucket=bucket, metadata=metadata)

    docs = [
        d
        for d in load_directory(path.parent)
        if str(d.metadata.get("source_file")) == path.name
        or Path(str(d.metadata.get("source", ""))).resolve() == path.resolve()
    ]
    return ingest_documents(docs)


def reindex_file(file_path: Path | str) -> IngestionReport:
    """Delete old chunks for this file, then ingest again."""
    path = Path(file_path).resolve()
    get_vector_store().delete(where={"source_file": path.name})
    return ingest_file(path, copy_into_bucket=False)


def build_vector_database(*, reset: bool = False) -> IngestionReport:
    if reset:
        reset_vector_store_singleton()
        reset_bm25_singleton()
        get_vector_store().reset()

    documents_config.uploads_path.mkdir(parents=True, exist_ok=True)
    return ingest_documents(load_all_sources(documents_config))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG ingestion pipeline")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument(
        "--bucket",
        default="uploads",
        choices=["laws", "regulations", "internal_docs", "uploads"],
    )
    args = parser.parse_args()

    report = (
        ingest_file(args.file, bucket=args.bucket)
        if args.file
        else build_vector_database(reset=args.reset)
    )
    print(
        f"files={len(report.files_indexed)} chunks={report.total_chunks} "
        f"vectors={report.vector_count}"
    )
    for name, count in sorted(report.files_indexed.items()):
        print(f"  - {name}: {count}")
