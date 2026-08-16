"""Copy a source file into a corpus bucket and write sidecar metadata."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal, Optional

from RAG.configuration.rag_config_loader import documents_config
from RAG.metadata.registry import save_source_metadata
from RAG.metadata.schema import SourceMetadata

SourceBucket = Literal["laws", "regulations", "internal_docs", "uploads"]


def upload_file(
    source_path: Path | str,
    *,
    bucket: SourceBucket = "uploads",
    metadata: Optional[SourceMetadata] = None,
    overwrite: bool = True,
) -> Path:
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    dest_dir = {
        "laws": documents_config.laws_path,
        "regulations": documents_config.regulations_path,
        "internal_docs": documents_config.internal_docs_path,
        "uploads": documents_config.uploads_path,
    }[bucket]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source_path.name
    if dest.exists() and not overwrite:
        raise FileExistsError(dest)

    shutil.copy2(source_path, dest)
    meta = metadata or SourceMetadata(law_name=dest.stem, source_type=bucket)
    if meta.source_type == "unknown":
        meta.source_type = bucket
    save_source_metadata(dest, meta)
    return dest
