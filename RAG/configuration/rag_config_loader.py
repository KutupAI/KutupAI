"""
rag_config_loader.py
----------------------
Türü güvenli RAG ayar yükleyicisi.

Tüm deneysel çalışma değerleri ``rag_config.yaml`` dosyasından okunur. Böylece
teslim alan ekip, kaynak kodu değiştirmeden aynı benchmark'ı tekrar çalıştırır.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
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


def _resolve_device(value: object) -> str:
    """Use CUDA when available without making GPU availability a startup requirement."""
    requested = str(value or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class DocumentsConfig:
    laws_path: Path
    regulations_path: Path
    amendments_path: Path
    internal_docs_path: Path
    uploads_path: Path
    reference_docs_path: Path
    classification_data_path: Path
    include_classification_text_data: bool
    text_globs: Tuple[str, ...]
    pdf_globs: Tuple[str, ...]
    docx_globs: Tuple[str, ...]
    spreadsheet_globs: Tuple[str, ...]

    @property
    def all_source_dirs(self) -> List[Path]:
        directories = [
            self.laws_path,
            self.regulations_path,
            self.amendments_path,
            self.internal_docs_path,
            self.uploads_path,
            self.reference_docs_path,
        ]
        if self.include_classification_text_data:
            directories.append(self.classification_data_path)
        return directories


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    separators: Tuple[str, ...]


@dataclass(frozen=True)
class IndexingConfig:
    batch_size: int
    ingest_batch_size: int
    reindex_on_update: bool


@dataclass(frozen=True)
class EmbeddingRuntimeConfig:
    """Embedding modelinin model, normalizasyon ve cihaz-batch ayarları."""

    model_name: str
    embedding_dim: int
    normalize_embeddings: bool
    batch_size_cpu: int
    batch_size_cuda: int
    show_progress: bool


@dataclass(frozen=True)
class VectorStoreConfig:
    """Yerel kalıcı Chroma koleksiyonunun ayarları."""

    persist_directory: Path
    collection_name: str
    distance_metric: str


@dataclass(frozen=True)
class RetrievalConfig:
    default_top_k: int
    max_top_k: int
    candidate_k: int
    max_candidate_k: int
    mode: str
    rrf_k: int
    vector_weight: float
    bm25_weight: float
    min_final_score: float


@dataclass(frozen=True)
class ObservabilityConfig:
    """İnteraktif retrieval ekranındaki isteğe bağlı tanı çıktıları."""

    retrieval_debug: bool
    show_stage_timings: bool
    show_query_details: bool
    show_candidate_details: bool
    show_result_metadata: bool


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
    base_rank_weight: float


@dataclass(frozen=True)
class AmendmentLedgerConfig:
    enabled: bool
    output_path: Path
    max_results: int


@dataclass(frozen=True)
class FactsRegistryConfig:
    enabled: bool
    output_path: Path
    max_results: int
    collection_name: str


@dataclass(frozen=True)
class LegalIndexConfig:
    """Yerel SQLite metadata/FTS indeksinin çalışma ayarları."""

    enabled: bool
    output_path: Path
    fts_candidate_k: int


@dataclass(frozen=True)
class MultiHopConfig:
    enabled: bool
    max_subqueries: int


@dataclass(frozen=True)
class QueryExpansionConfig:
    enabled: bool
    strategies: Tuple[str, ...]
    selected_strategy: str
    max_extra_terms: int


@dataclass(frozen=True)
class QueryTransformConfig:
    enabled: bool
    use_llm: bool
    base_url: str
    max_queries: int
    max_query_chars: int
    max_tokens: int
    timeout_seconds: int


@dataclass(frozen=True)
class AgentConfig:
    """Kaynaklı cevap üreten yerel LLM istemcisinin güvenli sınırları."""

    base_url: str
    timeout_seconds: int
    temperature: float
    top_p: float
    max_tokens: int
    context_max_tokens: int
    max_chunk_chars: int


@dataclass(frozen=True)
class CacheConfig:
    """Yerel semantic cache kapasitesi ve benzerlik eşiği."""

    enabled: bool
    threshold: float
    max_entries: int


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
    amendments_path=resolve_path(_raw["documents"].get("amendments_path", "RAG/documents/amendments")),
    internal_docs_path=resolve_path(_raw["documents"]["internal_docs_path"]),
    uploads_path=resolve_path(_raw["documents"].get("uploads_path", "RAG/documents/uploads")),
    reference_docs_path=resolve_path(_raw["documents"].get("reference_docs_path", "RAG/documents/reference_docs")),
    classification_data_path=resolve_path(_raw["documents"].get("classification_data_path", "RAG/documents/classification_data")),
    include_classification_text_data=bool(_raw["documents"].get("include_classification_text_data", False)),
    text_globs=tuple(_raw["documents"]["text_globs"]),
    pdf_globs=tuple(_raw["documents"]["pdf_globs"]),
    docx_globs=tuple(_raw["documents"].get("docx_globs", ["**/*.docx"])),
    spreadsheet_globs=tuple(_raw["documents"].get("spreadsheet_globs", ["**/*.xlsx"])),
)

chunking_config = ChunkingConfig(
    chunk_size=int(_raw["chunking"]["chunk_size"]),
    chunk_overlap=int(_raw["chunking"]["chunk_overlap"]),
    separators=tuple(_raw["chunking"]["separators"]),
)

indexing_config = IndexingConfig(
    batch_size=int(_raw["indexing"]["batch_size"]),
    ingest_batch_size=int(_raw["indexing"].get("ingest_batch_size", 384)),
    reindex_on_update=bool(_raw["indexing"]["reindex_on_update"]),
)

_emb = _raw.get("embedding", {})
embedding_runtime_config = EmbeddingRuntimeConfig(
    model_name=str(_emb.get("model_name", "BAAI/bge-m3")),
    embedding_dim=int(_emb.get("embedding_dim", 1024)),
    normalize_embeddings=bool(_emb.get("normalize_embeddings", True)),
    batch_size_cpu=max(1, int(_emb.get("batch_size_cpu", 32))),
    batch_size_cuda=max(1, int(_emb.get("batch_size_cuda", 64))),
    show_progress=bool(_emb.get("show_progress", True)),
)

_store = _raw.get("vector_store", {})
vector_store_config = VectorStoreConfig(
    persist_directory=resolve_path(str(_store.get("persist_directory", "RAG/documents/.chroma_db"))),
    collection_name=str(_store.get("collection_name", "legal_documents")),
    distance_metric=str(_store.get("distance_metric", "cosine")),
)

_ret = _raw["retrieval"]
retrieval_config = RetrievalConfig(
    default_top_k=int(_ret["default_top_k"]),
    max_top_k=int(_ret["max_top_k"]),
    candidate_k=int(_ret.get("candidate_k", 20)),
    max_candidate_k=int(_ret.get("max_candidate_k", 40)),
    mode=str(_ret.get("mode", "hybrid")),
    rrf_k=int(_ret.get("rrf_k", 60)),
    vector_weight=float(_ret.get("vector_weight", 0.55)),
    bm25_weight=float(_ret.get("bm25_weight", 0.45)),
    min_final_score=float(_ret.get("min_final_score", 0.05)),
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
    base_rank_weight=min(1.0, max(0.0, float(_rr.get("base_rank_weight", 0.0)))),
)

_ledger = _raw.get("amendment_ledger", {})
amendment_ledger_config = AmendmentLedgerConfig(
    enabled=bool(_ledger.get("enabled", True)),
    output_path=resolve_path(str(_ledger.get("output_path", "RAG/documents/amendment_ledger.json"))),
    max_results=max(1, int(_ledger.get("max_results", 4))),
)

_facts = _raw.get("facts_registry", {})
facts_registry_config = FactsRegistryConfig(
    enabled=bool(_facts.get("enabled", True)),
    output_path=resolve_path(str(_facts.get("output_path", "RAG/documents/facts_registry.json"))),
    max_results=max(1, int(_facts.get("max_results", 3))),
    collection_name=str(_facts.get("collection_name", "legal_facts")),
)

_legal_index = _raw.get("legal_index", {})
legal_index_config = LegalIndexConfig(
    enabled=bool(_legal_index.get("enabled", True)),
    output_path=resolve_path(str(_legal_index.get("output_path", "RAG/documents/.legal_index.sqlite"))),
    fts_candidate_k=max(1, int(_legal_index.get("fts_candidate_k", 20))),
)

_multi_hop = _raw.get("multi_hop", {})
multi_hop_config = MultiHopConfig(
    enabled=bool(_multi_hop.get("enabled", True)),
    max_subqueries=max(1, int(_multi_hop.get("max_subqueries", 2))),
)

_obs = _raw.get("observability", {})
observability_config = ObservabilityConfig(
    retrieval_debug=bool(_obs.get("retrieval_debug", True)),
    show_stage_timings=bool(_obs.get("show_stage_timings", True)),
    show_query_details=bool(_obs.get("show_query_details", True)),
    show_candidate_details=bool(_obs.get("show_candidate_details", True)),
    show_result_metadata=bool(_obs.get("show_result_metadata", True)),
)

_qe = _raw.get("query_expansion", {})
query_expansion_config = QueryExpansionConfig(
    enabled=bool(_qe.get("enabled", False)),
    strategies=tuple(_qe.get("strategies", ["none", "prf", "synonym"])),
    selected_strategy=str(_qe.get("selected_strategy", "none")),
    max_extra_terms=int(_qe.get("max_extra_terms", 4)),
)

_qt = _raw.get("query_transform", {})
query_transform_config = QueryTransformConfig(
    # Benchmark ve üretim, kayıtlı ayar dosyasını değiştirmeden bu değeri
    # ortam değişkeniyle açıkça geçersiz kılabilir.
    enabled=os.getenv("RAG_QUERY_TRANSFORM_ENABLED", str(_qt.get("enabled", False))).strip().lower()
    in {"1", "true", "yes", "on"},
    use_llm=os.getenv("RAG_QUERY_TRANSFORM_USE_LLM", str(_qt.get("use_llm", False))).strip().lower()
    in {"1", "true", "yes", "on"},
    base_url=os.getenv("RAG_QUERY_TRANSFORM_BASE_URL", str(_qt.get("base_url", "http://127.0.0.1:8081/v1/chat/completions"))).strip(),
    max_queries=max(1, int(_qt.get("max_queries", 3))),
    max_query_chars=max(32, int(_qt.get("max_query_chars", 320))),
    max_tokens=max(32, int(_qt.get("max_tokens", 96))),
    timeout_seconds=max(1, int(_qt.get("timeout_seconds", 8))),
)

_agent = _raw.get("agent", {})
agent_config = AgentConfig(
    base_url=str(_agent.get("base_url", "http://127.0.0.1:8080/v1/chat/completions")),
    timeout_seconds=max(1, int(_agent.get("timeout_seconds", 45))),
    temperature=float(_agent.get("temperature", 0.0)),
    top_p=float(_agent.get("top_p", 1.0)),
    max_tokens=max(1, int(_agent.get("max_tokens", 450))),
    context_max_tokens=max(256, int(_agent.get("context_max_tokens", 2700))),
    max_chunk_chars=max(200, int(_agent.get("max_chunk_chars", 1500))),
)

_cache = _raw.get("cache", {})
cache_config = CacheConfig(
    enabled=bool(_cache.get("enabled", True)),
    threshold=min(1.0, max(-1.0, float(_cache.get("threshold", 0.94)))),
    max_entries=max(1, int(_cache.get("max_entries", 200))),
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

runtime_config = RuntimeConfig(device=_resolve_device(_raw.get("runtime", {}).get("device", "auto")))
