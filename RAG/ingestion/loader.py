"""Load corpus files with DirectoryLoader + TextLoader / PyPDFLoader."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from RAG.configuration.rag_config_loader import DocumentsConfig, documents_config

_SKIP_NAMES = {"readme.md", "readme.txt", ".gitkeep"}


def _load_with(directory: Path, globs: tuple[str, ...], loader_cls, loader_kwargs=None) -> List[Document]:
    if not directory.exists():
        return []
    docs: List[Document] = []
    for pattern in globs:
        loader = DirectoryLoader(
            path=str(directory),
            glob=pattern,
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs or {},
            show_progress=False,
            use_multithreading=False,
            silent_errors=True,
        )
        docs.extend(loader.load())
    return docs


def _bucket_name(directory: Path, cfg: DocumentsConfig) -> str:
    resolved = directory.resolve()
    mapping = {
        cfg.laws_path.resolve(): "laws",
        cfg.regulations_path.resolve(): "regulations",
        cfg.internal_docs_path.resolve(): "internal_docs",
        cfg.uploads_path.resolve(): "uploads",
    }
    return mapping.get(resolved, "unknown")


def _keep(path: Path) -> bool:
    name = path.name.lower()
    return name not in _SKIP_NAMES and not name.startswith(".") and not name.endswith(".meta.json")


def load_directory(directory: Path, cfg: DocumentsConfig | None = None) -> List[Document]:
    cfg = cfg or documents_config
    raw = _load_with(directory, cfg.text_globs, TextLoader, {"encoding": "utf-8"})
    raw += _load_with(directory, cfg.pdf_globs, PyPDFLoader)

    source_type = _bucket_name(directory, cfg)
    kept: List[Document] = []
    for doc in raw:
        meta = dict(doc.metadata or {})
        source = Path(str(meta.get("source", "")))
        if not _keep(source):
            continue
        meta.update(
            {
                "source_type": source_type,
                "source_dir": str(directory),
                "source_file": source.name or "unknown",
                "law_name": meta.get("law_name") or source.stem or "unknown",
            }
        )
        doc.metadata = meta
        kept.append(doc)
    return kept


def load_all_sources(cfg: DocumentsConfig | None = None) -> List[Document]:
    cfg = cfg or documents_config
    docs: List[Document] = []
    for directory in cfg.all_source_dirs:
        docs.extend(load_directory(directory, cfg))
    return docs
