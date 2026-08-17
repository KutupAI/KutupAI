"""Corpus-version manifest for safe incremental indexing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from langchain_core.documents import Document


MANIFEST_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CorpusDiff:
    changed_sources: List[str]
    removed_sources: List[str]


class CorpusManifest:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.entries: Dict[str, Dict[str, str]] = {}

    def load(self) -> None:
        if not self.path.exists():
            self.entries = {}
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("version") != MANIFEST_VERSION:
            raise ValueError(f"Unsupported manifest version in {self.path}")
        self.entries = dict(payload.get("sources") or {})

    def diff(self, documents: Iterable[Document]) -> CorpusDiff:
        current: Dict[str, str] = {}
        for document in documents:
            source = Path(str(document.metadata.get("source", ""))).resolve()
            source_file = str(document.metadata.get("source_file") or source.name)
            if source.exists() and source_file not in current:
                current[source_file] = sha256_file(source)
        changed = sorted(
            name for name, checksum in current.items()
            if self.entries.get(name, {}).get("sha256") != checksum
        )
        return CorpusDiff(
            changed_sources=changed,
            removed_sources=sorted(set(self.entries) - set(current)),
        )

    def write(self, documents: Iterable[Document]) -> None:
        sources: Dict[str, Dict[str, str]] = {}
        for document in documents:
            source = Path(str(document.metadata.get("source", ""))).resolve()
            source_file = str(document.metadata.get("source_file") or source.name)
            if source.exists() and source_file not in sources:
                sources[source_file] = {
                    "sha256": sha256_file(source),
                    "source_path": str(source),
                    "source_type": str(document.metadata.get("source_type", "unknown")),
                }
        payload = {
            "version": MANIFEST_VERSION,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
