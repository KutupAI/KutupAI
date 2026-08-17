"""BM25 lexical index (persisted next to Chroma)."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from rank_bm25 import BM25Okapi

from RAG.chroma.chroma_config import chroma_config
from RAG.retriever.text_utils import tokenize
from RAG.vector_store.vector_store_interface import SearchResult


@dataclass
class Bm25Document:
    chunk_id: str
    text: str
    metadata: dict


class Bm25Index:
    def __init__(self, documents: Sequence[Bm25Document] | None = None) -> None:
        self.docs: List[Bm25Document] = list(documents or [])
        self._bm25: Optional[BM25Okapi] = None
        if self.docs:
            corpus = [tokenize(d.text) or ["_empty_"] for d in self.docs]
            self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10, where: dict | None = None) -> List[SearchResult]:
        if not self._bm25 or not self.docs or not query.strip():
            return []
        scores = self._bm25.get_scores(tokenize(query))

        def matches(metadata: dict) -> bool:
            if not where:
                return True
            for key, expected in where.items():
                actual = metadata.get(key)
                if isinstance(expected, dict) and "$in" in expected:
                    if actual not in expected["$in"]:
                        return False
                elif actual != expected:
                    return False
            return True

        # Filtreleme sıralamadan önce yapılır. Global kısa listeyi sonradan
        # filtrelemek doğru maddeyi aday olmadan eleyebilir.
        eligible = [i for i, doc in enumerate(self.docs) if matches(doc.metadata)]
        ranked = sorted(eligible, key=lambda i: scores[i], reverse=True)[:top_k]
        max_score = max((scores[i] for i in ranked), default=1.0) or 1.0
        out: List[SearchResult] = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            doc = self.docs[i]
            out.append(
                SearchResult(
                    id=doc.chunk_id,
                    text=doc.text,
                    metadata=dict(doc.metadata),
                    score=round(float(scores[i]) / float(max_score), 4),
                )
            )
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.docs, f)

    @classmethod
    def load(cls, path: Path) -> "Bm25Index":
        if not path.exists():
            return cls([])
        with open(path, "rb") as f:
            return cls(pickle.load(f))


def bm25_path() -> Path:
    return Path(chroma_config.persist_directory) / "bm25_index.pkl"


_index: Optional[Bm25Index] = None


def get_bm25_index() -> Bm25Index:
    global _index
    if _index is None:
        _index = Bm25Index.load(bm25_path())
    return _index


def rebuild_bm25_from_chunks(chunks: Sequence[dict]) -> Bm25Index:
    global _index
    docs = [
        Bm25Document(str(c["chunk_id"]), str(c["text"]), dict(c.get("metadata") or {}))
        for c in chunks
    ]
    _index = Bm25Index(docs)
    _index.save(bm25_path())
    return _index


def reset_bm25_singleton() -> None:
    global _index
    _index = None
