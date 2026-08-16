"""Unified retrieval response for Agents."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class RetrievalResponse:
    context: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    result_count: int = 0
