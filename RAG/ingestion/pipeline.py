"""Ingestion: upload → load → chunk → metadata → Chroma + BM25."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from RAG.configuration.rag_config_loader import documents_config, indexing_config
from RAG.ingestion.chunker import split_documents
from RAG.ingestion.enricher import enrich_documents
from RAG.ingestion.loader import load_all_sources, load_directory
from RAG.ingestion.manifest import CorpusManifest
from RAG.ingestion.uploader import SourceBucket, upload_file
from RAG.metadata.registry import apply_source_metadata
from RAG.metadata.schema import SourceMetadata, validate_chunk_metadata
from RAG.retriever.bm25_index import rebuild_bm25_from_chunks, reset_bm25_singleton
from RAG.vector_store.chroma_store import get_vector_store, reset_vector_store_singleton


def _invalidate_answer_cache() -> None:
    """Corpus changes invalidate generated answers even if a query is identical."""
    from RAG.agent.semantic_cache import SemanticCache
    from RAG.graph.legal_graph import reset_legal_graph

    SemanticCache().clear()
    reset_legal_graph()


@dataclass
class IngestionReport:
    files_indexed: Dict[str, int]
    total_chunks: int
    vector_count: int
    invalid_metadata: int = 0


def _prepare(documents: List[Document]) -> List[Document]:
    return enrich_documents(split_documents(apply_source_metadata(documents)))


def _unique_chunks(chunks: List[Document]) -> List[Document]:
    """Keep one deterministic copy of each chunk before every side effect."""
    unique: List[Document] = []
    seen_ids = set()
    for chunk in chunks:
        chunk_id = str(chunk.metadata["chunk_id"])
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique.append(chunk)
    return unique


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
    pending: List[Document] = []
    seen_ids = set()
    total_chunks = 0
    export_path = Path("RAG/documents/indexed_chunks.json")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_export = export_path.with_suffix(".json.tmp")
    export_file = temporary_export.open("w", encoding="utf-8")
    export_file.write("[")
    first_export = True

    def flush() -> None:
        """Persist a bounded chunk batch, keeping large corpora memory-safe."""
        nonlocal invalid, total_chunks, first_export
        if not pending:
            return
        invalid += _upsert(pending)
        for chunk in pending:
            if not first_export:
                export_file.write(",\n")
            json.dump(_chunk_export_record(chunk), export_file, ensure_ascii=False)
            first_export = False
        total_chunks += len(pending)
        pending.clear()

    try:
        for name, docs in by_file.items():
            for chunk in _prepare(docs):
                chunk_id = str(chunk.metadata["chunk_id"])
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                pending.append(chunk)
                summary[name] = summary.get(name, 0) + 1
                if len(pending) >= indexing_config.ingest_batch_size:
                    flush()
        flush()
        export_file.write("]\n")
    finally:
        export_file.close()
    try:
        # Atomik değiştirme, yarım yazılmış bir inceleme dosyası bırakmamak
        # için normal yoldur. Windows'ta VS Code/Explorer dosyayı açık
        # tutabiliyor; bu yardımcı çıktı yüzünden başarılı indeksleme durmaz.
        temporary_export.replace(export_path)
    except PermissionError:
        fallback_export = export_path.with_name(f"{export_path.stem}_latest.json")
        temporary_export.replace(fallback_export)
        export_path = fallback_export
        print(
            "Warning: indexed_chunks.json is open in another application; "
            f"the current inspection export was written to {export_path}."
        )

    if rebuild_bm25:
        rebuild_bm25_from_chunks(get_vector_store().export_all())

    print(f"Exported chunks for inspection: {export_path} ({total_chunks} chunks)")

    return IngestionReport(
        files_indexed=summary,
        total_chunks=total_chunks,
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
    report = ingest_file(path, copy_into_bucket=False)
    _invalidate_answer_cache()
    return report

def _chunk_export_record(doc: Document) -> dict:
    """Return the compact, human-inspection representation of one chunk."""
    return {
        "chunk_id": doc.metadata.get("chunk_id", "unknown"),
        "law_number": doc.metadata.get("law_number", "unknown"),
        "article_no": doc.metadata.get("article_no"),
        "article_type": doc.metadata.get("article_type", "unknown"),
        "source_file": doc.metadata.get("source_file", "unknown"),
        "page_start": doc.metadata.get("page_start", doc.metadata.get("page")),
        "page_end": doc.metadata.get("page_end", doc.metadata.get("page")),
        "text_length": len(doc.page_content),
        "text_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
        "full_text": doc.page_content,
    }


def _export_chunks_to_json(documents: List[Document], output_path: str = "RAG/documents/indexed_chunks.json"):
    """
    🚀 KutupAI Özelliği: Tüm chunk'ları incelemek ve debug etmek için 
    okunabilir bir JSON dosyasına kaydeder.
    """
    export_data = []
    for doc in documents:
        export_data.append({
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "law_number": doc.metadata.get("law_number", "unknown"),
            "article_no": doc.metadata.get("article_no"),
            "article_type": doc.metadata.get("article_type", "unknown"),
            "source_file": doc.metadata.get("source_file", "unknown"),
            "page_start": doc.metadata.get("page_start", doc.metadata.get("page")),
            "page_end": doc.metadata.get("page_end", doc.metadata.get("page")),
            "text_length": len(doc.page_content),
            "text_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content, # İlk 300 karakter önizleme
            "full_text": doc.page_content # Tam metin (Dosya boyutu çok büyük olursa bunu kaldırabilirsiniz)
        })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
        
    print(f"Exported chunks for inspection: {output_path} ({len(export_data)} chunks)")


def build_vector_database(*, reset: bool = False) -> IngestionReport:
    if reset:
        reset_vector_store_singleton()
        reset_bm25_singleton()
        get_vector_store().reset()
        _invalidate_answer_cache()

    documents_config.uploads_path.mkdir(parents=True, exist_ok=True)
    return ingest_documents(load_all_sources(documents_config))


def sync_vector_database() -> IngestionReport:
    """Index only modified sources and remove chunks for deleted sources.

    The manifest is updated only after the vector and BM25 stores complete
    successfully, so an interrupted run remains safe to retry.
    """
    documents = load_all_sources(documents_config)
    manifest = CorpusManifest(documents_config.laws_path.parent / "manifest.json")
    manifest.load()
    diff = manifest.diff(documents)
    changed = set(diff.changed_sources)

    for source_file in diff.removed_sources:
        get_vector_store().delete(where={"source_file": source_file})
    selected = [doc for doc in documents if str(doc.metadata.get("source_file")) in changed]
    for source_file in changed:
        # İçeriğe bağlı kimlikler kaynak değişince eski chunk'ları kendiliğinden
        # kaldırmaz; bu nedenle upsert öncesinde eski kaynak kaydı silinir.
        get_vector_store().delete(where={"source_file": source_file})

    if selected:
        report = ingest_documents(selected, rebuild_bm25=False)
    else:
        report = IngestionReport({}, 0, get_vector_store().count())
    if selected or diff.removed_sources:
        rebuild_bm25_from_chunks(get_vector_store().export_all())
        _invalidate_answer_cache()
    manifest.write(documents)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG ingestion pipeline")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--incremental", action="store_true", help="Index only changed/deleted corpus files.")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument(
        "--bucket",
        default="uploads",
        choices=["laws", "regulations", "amendments", "internal_docs", "uploads"],
    )
    args = parser.parse_args()

    if args.file and args.incremental:
        parser.error("--file and --incremental cannot be used together")
    if args.reset and args.incremental:
        parser.error("--reset and --incremental cannot be used together")
    report = (
        ingest_file(args.file, bucket=args.bucket)
        if args.file
        else sync_vector_database()
        if args.incremental
        else build_vector_database(reset=args.reset)
    )
    print(
        f"files={len(report.files_indexed)} chunks={report.total_chunks} "
        f"vectors={report.vector_count}"
    )
    for name, count in sorted(report.files_indexed.items()):
        # CP1252 kullanan Windows konsolları bazı Türkçe harfleri yazamaz.
        # Başarılı ingestion'ın CLI hatası gibi görünmesi önlenir.
        safe_line = f"  - {name}: {count}".encode("cp1252", errors="replace").decode("cp1252")
        print(safe_line)
