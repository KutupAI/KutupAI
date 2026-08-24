"""Small persistent embedding-based semantic cache for complete legal answers."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from RAG.embeddings.embedding_model import embed_text
from RAG.configuration.rag_config_loader import cache_config

# Retrieval, context, citation veya prompt değiştiğinde artırılır. Eski cevaplar
# akıcı görünse de farklı bir kanıt hattından geldiği için güvenlik düzeltmesini
# aşmamalıdır.
CACHE_SCHEMA_VERSION = 24


@dataclass(frozen=True)
class CacheHit:
    payload: Dict[str, Any]
    similarity: float
    exact: bool


class SemanticCache:
    """JSON-backed cache; suitable for a single-machine local RAG deployment.

    A shared multi-worker deployment should replace this storage adapter with
    Redis, without changing the answer-agent interface.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        threshold: float = cache_config.threshold,
        max_entries: int = cache_config.max_entries,
    ) -> None:
        self.path = path or Path(__file__).resolve().parents[1] / "documents" / ".semantic_cache.json"
        self.threshold = threshold
        self.max_entries = max_entries

    @staticmethod
    def _normalise(query: str) -> str:
        return " ".join((query or "").casefold().split())

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _save(self, rows: List[Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(rows[-self.max_entries :], ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _cosine(left: List[float], right: List[float]) -> float:
        if not left or len(left) != len(right):
            return -1.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        return dot / (left_norm * right_norm) if left_norm and right_norm else -1.0

    def get(self, query: str) -> Optional[CacheHit]:
        normalised = self._normalise(query)
        rows = self._load()
        if not rows:
            return None
        for row in reversed(rows):
            if row.get("version") == CACHE_SCHEMA_VERSION and row.get("query") == normalised:
                return CacheHit(payload=dict(row.get("payload") or {}), similarity=1.0, exact=True)

        vector = list(map(float, embed_text(query)))
        best: Optional[CacheHit] = None
        for row in rows:
            if row.get("version") != CACHE_SCHEMA_VERSION:
                continue
            score = self._cosine(vector, list(map(float, row.get("embedding") or [])))
            if score >= self.threshold and (best is None or score > best.similarity):
                best = CacheHit(payload=dict(row.get("payload") or {}), similarity=round(score, 6), exact=False)
        return best

    def put(self, query: str, payload: Dict[str, Any]) -> None:
        rows = self._load()
        normalised = self._normalise(query)
        rows = [row for row in rows if row.get("query") != normalised]
        rows.append(
            {
                "query": normalised,
                "version": CACHE_SCHEMA_VERSION,
                "embedding": list(map(float, embed_text(query))),
                "payload": payload,
                "created_at": int(time.time()),
            }
        )
        self._save(rows)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
