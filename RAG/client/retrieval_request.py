"""Retrieval request shape used by Agents."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: Optional[int] = None
    mode: Optional[str] = None  # hybrid | vector | bm25
    use_prf: Optional[bool] = None
    use_reranker: Optional[bool] = None
    expansion_strategy: Optional[str] = None
    source_type: Optional[str] = None
