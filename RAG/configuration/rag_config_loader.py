"""
rag_config_loader.py
-----------------------
يقرأ rag_config.yaml ويحوّل قيمه إلى كائنات Python جاهزة للاستخدام
(بنفس أسلوب embedding_config.py و chroma_config.py).

هذا هو الملف الوحيد الذي يقرأ rag_config.yaml مباشرة؛ باقي الملفات
تستورد chunking_config / retrieval_config / documents_config من هنا.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

_CONFIG_PATH = Path(__file__).parent / "rag_config.yaml"


def _load_raw_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_raw = _load_raw_config()


@dataclass(frozen=True)
class DocumentsConfig:
    laws_path: str
    regulations_path: str
    internal_docs_path: str
    allowed_extensions: List[str]


@dataclass(frozen=True)
class ChunkingConfig:
    unit: str
    max_chunk_size_chars: int
    chunk_overlap_chars: int


@dataclass(frozen=True)
class IndexingConfig:
    batch_size: int
    reindex_on_update: bool


@dataclass(frozen=True)
class RetrievalConfig:
    default_top_k: int
    max_top_k: int


documents_config = DocumentsConfig(
    laws_path=_raw["documents"]["laws_path"],
    regulations_path=_raw["documents"]["regulations_path"],
    internal_docs_path=_raw["documents"]["internal_docs_path"],
    allowed_extensions=_raw["documents"]["allowed_extensions"],
)

chunking_config = ChunkingConfig(
    unit=_raw["chunking"]["unit"],
    max_chunk_size_chars=_raw["chunking"]["max_chunk_size_chars"],
    chunk_overlap_chars=_raw["chunking"]["chunk_overlap_chars"],
)

indexing_config = IndexingConfig(
    batch_size=_raw["indexing"]["batch_size"],
    reindex_on_update=_raw["indexing"]["reindex_on_update"],
)

retrieval_config = RetrievalConfig(
    default_top_k=_raw["retrieval"]["default_top_k"],
    max_top_k=_raw["retrieval"]["max_top_k"],
)
