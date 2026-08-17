"""Compatibility API for the supported ingestion pipeline.

New callers should import :mod:`RAG.ingestion.pipeline` directly. This module
keeps older imports working without maintaining a second incompatible indexer.
"""

from pathlib import Path

from RAG.ingestion.pipeline import build_vector_database, ingest_directory, reindex_file


def index_file(file_path: Path) -> int:
    """Reindex one supported source file and return its resulting chunk count."""
    return reindex_file(file_path).total_chunks


def index_directory(directory_path: Path) -> dict[str, int]:
    """Index a directory through the canonical ingestion implementation."""
    return ingest_directory(directory_path).files_indexed


def index_all_sources() -> dict[str, int]:
    """Rebuild every configured corpus source through the canonical pipeline."""
    return build_vector_database().files_indexed


if __name__ == "__main__":
    result = index_all_sources()
    total_chunks = sum(result.values())
    print(f"Indexed {len(result)} files, {total_chunks} chunks.")
