"""Metadata system: schema + sidecar registry."""

from RAG.metadata.registry import (
    apply_source_metadata,
    load_source_metadata,
    meta_path_for,
    save_source_metadata,
)
from RAG.metadata.schema import ChunkMetadata, SourceMetadata, validate_chunk_metadata

__all__ = [
    "ChunkMetadata",
    "SourceMetadata",
    "validate_chunk_metadata",
    "apply_source_metadata",
    "load_source_metadata",
    "save_source_metadata",
    "meta_path_for",
]
