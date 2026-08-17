"""İndekslenmiş corpus için bellek içi hukukî Graph-RAG zenginleştirmesi.

Graf, vektör veritabanını tek doğruluk kaynağı tutar. İkinci bir embedding
indeksi üretmez; yalnız güvenilir kanun/madde dayanağı olan sorularda reranker
adaylarına izlenebilir komşu maddeler ekler.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Set, Tuple

from RAG.retriever.query_metadata import get_query_metadata_extractor
from RAG.vector_store.chroma_store import get_vector_store
from RAG.vector_store.vector_store_interface import SearchResult


# Kanunlar arası ilişki hem kanun hem madde gerektirir. Başka bir kanunun
# rastgele ilk maddesine bağlanmak yanıltıcı hukukî ilişki oluşturur.
_CROSS_REFERENCE = re.compile(
    r"\b(?P<law>\d{3,4})\s*(?:sayılı|sayili)\b.{0,140}?"
    r"(?:(?P<article1>\d+)\s*\.?\s*madd(?:e|esi|esine|esinin)|madde\s*(?P<article2>\d+))",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class GraphStats:
    laws: int
    articles: int
    adjacent_edges: int
    reference_edges: int = 0


class LegalKnowledgeGraph:
    """Law/article graph with same-law and explicit cross-law references."""

    def __init__(self) -> None:
        self._by_law: Dict[str, List[str]] = defaultdict(list)
        self._article_chunks: Dict[str, List[str]] = defaultdict(list)
        self._chunks: Dict[str, SearchResult] = {}
        self._neighbors: Dict[str, Set[str]] = defaultdict(set)
        self._reference_neighbors: Dict[str, Set[str]] = defaultdict(set)
        self._built = False

    @staticmethod
    def _article_node(law_number: object, article_number: object) -> str:
        return f"law:{law_number}|article:{article_number}"

    @staticmethod
    def _metadata_node(metadata: Dict[str, Any]) -> str | None:
        law = str(metadata.get("law_number") or "").strip()
        article = str(metadata.get("article_no") or metadata.get("article_number") or "").strip()
        return LegalKnowledgeGraph._article_node(law, article) if law and article else None

    def build(self, rows: Iterable[Dict[str, object]] | None = None) -> GraphStats:
        rows = rows if rows is not None else get_vector_store().export_all()
        self._by_law.clear()
        self._article_chunks.clear()
        self._chunks.clear()
        self._neighbors.clear()
        self._reference_neighbors.clear()

        pending_references: List[Tuple[str, str, str]] = []
        for row in rows:
            meta = dict(row.get("metadata") or {})
            node = self._metadata_node(meta)
            chunk_id = str(row.get("chunk_id") or meta.get("chunk_id") or "")
            if not node or not chunk_id:
                continue
            law = str(meta.get("law_number"))
            if node not in self._by_law[law]:
                self._by_law[law].append(node)
            self._article_chunks[node].append(chunk_id)
            self._chunks[chunk_id] = SearchResult(
                id=chunk_id, text=str(row.get("text") or ""), metadata=meta, score=0.0
            )
            for match in _CROSS_REFERENCE.finditer(str(row.get("text") or "")):
                target_article = match.group("article1") or match.group("article2")
                if target_article:
                    pending_references.append((node, match.group("law"), target_article))

        adjacent_edges = 0
        for nodes in self._by_law.values():
            nodes.sort(key=lambda node: int(node.rsplit(":", 1)[-1]) if node.rsplit(":", 1)[-1].isdigit() else 10**9)
            for left, right in zip(nodes, nodes[1:]):
                self._neighbors[left].add(right)
                self._neighbors[right].add(left)
                adjacent_edges += 1

        reference_edges = 0
        for source, target_law, target_article in pending_references:
            target = self._article_node(target_law, target_article)
            if target in self._article_chunks:
                self._reference_neighbors[source].add(target)
                self._reference_neighbors[target].add(source)
                reference_edges += 1
        self._built = True
        return GraphStats(len(self._by_law), len(self._article_chunks), adjacent_edges, reference_edges)

    def related_chunk_ids(
        self, law_number: str, article_number: str, *, hops: int = 1, include_references: bool = False
    ) -> List[str]:
        if not self._built:
            self.build()
        start = self._article_node(law_number, article_number)
        distances = self._distances({start}, hops, include_references=include_references)
        return [chunk_id for node in distances for chunk_id in self._article_chunks.get(node, []) if chunk_id]

    def enrich(
        self, query: str, results: List[SearchResult], *, hops: int = 1, max_related: int = 12,
        include_references: bool = False,
    ) -> List[SearchResult]:
        """Append graph neighbours without hiding base retrieval hits.

        Expansion requires an explicit law and article, preventing broad
        questions from being pulled toward an accidental top-ranked law.
        """
        filters = get_query_metadata_extractor().extract(query)
        law, article = filters.get("law_number"), filters.get("article_no")
        if not law or not article:
            return results
        if not self._built:
            self.build()

        distances = self._distances(
            {self._article_node(law, article)}, hops, include_references=include_references
        )
        if not distances:
            return results

        # Kesin kanun/madde filtresi çoğunlukla yalnız istenen maddeyi getirir.
        # Aynı kanundaki komşular yüksek güvenli sonuçların ardından eklenir;
        # böylece küçük aday sınırında bile reranker onları değerlendirebilir.
        head = list(results[:8])
        tail = list(results[8:])
        combined = head
        seen = {item["id"] for item in results}
        added = 0
        for node, distance in sorted(distances.items(), key=lambda item: (item[1], item[0])):
            for chunk_id in self._article_chunks.get(node, []):
                if chunk_id in seen or added >= max_related:
                    continue
                original = self._chunks.get(chunk_id)
                if not original:
                    continue
                meta = dict(original["metadata"])
                meta.update({"graph_rag": True, "graph_node": node, "graph_distance": distance})
                combined.append(SearchResult(
                    id=original["id"], text=original["text"], metadata=meta,
                    score=round(max(0.05, 0.35 - (0.08 * distance)), 4),
                ))
                seen.add(chunk_id)
                added += 1
        return combined + tail

    def _distances(self, starts: Set[str], hops: int, *, include_references: bool = False) -> Dict[str, int]:
        distances: Dict[str, int] = {node: 0 for node in starts if node in self._article_chunks}
        queue = deque(distances)
        while queue:
            current = queue.popleft()
            if distances[current] >= max(0, hops):
                continue
            neighbors = set(self._neighbors.get(current, set()))
            if include_references:
                neighbors.update(self._reference_neighbors.get(current, set()))
            for neighbor in neighbors:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances


_graph_instance: LegalKnowledgeGraph | None = None


def get_legal_graph() -> LegalKnowledgeGraph:
    """Return one cached graph per process; reindexing starts a new process."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = LegalKnowledgeGraph()
    return _graph_instance


def reset_legal_graph() -> None:
    global _graph_instance
    _graph_instance = None
