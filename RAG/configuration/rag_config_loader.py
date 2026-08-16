"""
rag_config_loader.py
----------------------
Loads rag_config.yaml into typed config objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONFIG_DIR.parents[1]
_CONFIG_PATH = _CONFIG_DIR / "rag_config.yaml"


def _load_raw() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_raw = _load_raw()


def resolve_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class DocumentsConfig:
    laws_path: Path
    regulations_path: Path
    internal_docs_path: Path
    uploads_path: Path
    text_globs: Tuple[str, ...]
    pdf_globs: Tuple[str, ...]

    @property
    def all_source_dirs(self) -> List[Path]:
        return [
            self.laws_path,
            self.regulations_path,
            self.internal_docs_path,
            self.uploads_path,
        ]


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    separators: Tuple[str, ...]


@dataclass(frozen=True)
class IndexingConfig:
    batch_size: int
    reindex_on_update: bool


@dataclass(frozen=True)
class RetrievalConfig:
    default_top_k: int
    max_top_k: int
    candidate_k: int
    mode: str
    rrf_k: int
    vector_weight: float
    bm25_weight: float


@dataclass(frozen=True)
class PrfConfig:
    enabled: bool
    top_n_docs: int
    max_expand_terms: int
    min_term_len: int


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool
    model_name: str
    top_n: int


@dataclass(frozen=True)
class QueryExpansionConfig:
    enabled: bool
    strategies: Tuple[str, ...]
    selected_strategy: str
    max_extra_terms: int


@dataclass(frozen=True)
class EvaluationConfig:
    datasets_dir: Path
    default_dataset: Path
    hit_ks: Tuple[int, ...]
    max_questions_per_chunk: int
    llm_temperature: float
    llm_max_tokens: int


@dataclass(frozen=True)
class RuntimeConfig:
    device: str


documents_config = DocumentsConfig(
    laws_path=resolve_path(_raw["documents"]["laws_path"]),
    regulations_path=resolve_path(_raw["documents"]["regulations_path"]),
    internal_docs_path=resolve_path(_raw["documents"]["internal_docs_path"]),
    uploads_path=resolve_path(_raw["documents"].get("uploads_path", "RAG/documents/uploads")),
    text_globs=tuple(_raw["documents"]["text_globs"]),
    pdf_globs=tuple(_raw["documents"]["pdf_globs"]),
)

chunking_config = ChunkingConfig(
    chunk_size=int(_raw["chunking"]["chunk_size"]),
    chunk_overlap=int(_raw["chunking"]["chunk_overlap"]),
    separators=tuple(_raw["chunking"]["separators"]),
)

indexing_config = IndexingConfig(
    batch_size=int(_raw["indexing"]["batch_size"]),
    reindex_on_update=bool(_raw["indexing"]["reindex_on_update"]),
)

_ret = _raw["retrieval"]
retrieval_config = RetrievalConfig(
    default_top_k=int(_ret["default_top_k"]),
    max_top_k=int(_ret["max_top_k"]),
    candidate_k=int(_ret.get("candidate_k", 20)),
    mode=str(_ret.get("mode", "hybrid")),
    rrf_k=int(_ret.get("rrf_k", 60)),
    vector_weight=float(_ret.get("vector_weight", 0.55)),
    bm25_weight=float(_ret.get("bm25_weight", 0.45)),
)

_prf = _raw.get("prf", {})
prf_config = PrfConfig(
    enabled=bool(_prf.get("enabled", True)),
    top_n_docs=int(_prf.get("top_n_docs", 3)),
    max_expand_terms=int(_prf.get("max_expand_terms", 5)),
    min_term_len=int(_prf.get("min_term_len", 4)),
)

_rr = _raw.get("reranker", {})
reranker_config = RerankerConfig(
    enabled=bool(_rr.get("enabled", True)),
    model_name=str(_rr.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
    top_n=int(_rr.get("top_n", 15)),
)

_qe = _raw.get("query_expansion", {})
query_expansion_config = QueryExpansionConfig(
    enabled=bool(_qe.get("enabled", False)),
    strategies=tuple(_qe.get("strategies", ["none", "prf", "synonym"])),
    selected_strategy=str(_qe.get("selected_strategy", "none")),
    max_extra_terms=int(_qe.get("max_extra_terms", 4)),
)

_ev = _raw.get("evaluation", {})
_llm = _ev.get("llm_dataset", {})
evaluation_config = EvaluationConfig(
    datasets_dir=resolve_path(_ev.get("datasets_dir", "RAG/evaluation/datasets")),
    default_dataset=resolve_path(_ev.get("default_dataset", "RAG/evaluation/datasets/eval_set.json")),
    hit_ks=tuple(_ev.get("hit_ks", [1, 2, 3])),
    max_questions_per_chunk=int(_llm.get("max_questions_per_chunk", 2)),
    llm_temperature=float(_llm.get("temperature", 0.3)),
    llm_max_tokens=int(_llm.get("max_tokens", 512)),
)

runtime_config = RuntimeConfig(device=str(_raw["runtime"]["device"]))
