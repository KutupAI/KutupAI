"""Sidecar *.meta.json load/save + merge into Documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

from RAG.metadata.schema import SourceMetadata


def meta_path_for(source_file: Path) -> Path:
    return source_file.with_name(source_file.stem + ".meta.json")


def load_source_metadata(source_file: Path, default_source_type: str = "unknown") -> SourceMetadata:
    path = meta_path_for(source_file)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            meta = SourceMetadata.from_dict(json.load(f))
        if meta.source_type == "unknown":
            meta.source_type = default_source_type
        return meta
    return SourceMetadata(law_name=source_file.stem, source_type=default_source_type)


def save_source_metadata(source_file: Path, metadata: SourceMetadata) -> Path:
    path = meta_path_for(source_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def apply_source_metadata(
    documents: List[Document],
    *,
    default_source_type: str = "unknown",
) -> List[Document]:
    cache: Dict[str, SourceMetadata] = {}
    out: List[Document] = []
    for doc in documents:
        meta = dict(doc.metadata or {})
        source = Path(str(meta.get("source", meta.get("source_file", "unknown"))))
        key = str(source)
        if key not in cache:
            file_path = source if source.suffix else Path(str(meta.get("source_file", source.name)))
            cache[key] = load_source_metadata(
                file_path,
                default_source_type=str(meta.get("source_type", default_source_type)),
            )
        merged = cache[key].to_dict()
        for k, v in meta.items():
            if v not in (None, "", "unknown"):
                merged[k] = v
        if isinstance(merged.get("tags"), list):
            merged["tags"] = ",".join(str(t) for t in merged["tags"])
        out.append(Document(page_content=doc.page_content, metadata=merged))
    return out
