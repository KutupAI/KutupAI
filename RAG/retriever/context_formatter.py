"""Format retrieval hits for Agents (context text + sources)."""

from __future__ import annotations

from typing import Any, Dict, List

from RAG.vector_store.vector_store_interface import SearchResult


def format_context(results: List[SearchResult]) -> str:
    if not results:
        return "No relevant legal passages were found."
    parts = []
    for i, r in enumerate(results, start=1):
        m = r["metadata"]
        header = (
            f"[{m.get('law_name', 'unknown')} | Madde {m.get('article_number', '?')} "
            f"| {m.get('source_file', '?')} | score={r['score']}]"
        )
        parts.append(f"{i}. {header}\n{r['text']}")
    return "\n\n".join(parts)


def extract_sources(results: List[SearchResult]) -> List[Dict[str, Any]]:
    return [
        {
            "law_name": r["metadata"].get("law_name", "unknown"),
            "article_number": r["metadata"].get("article_number", "unknown"),
            "source_file": r["metadata"].get("source_file", "unknown"),
            "source_type": r["metadata"].get("source_type", "unknown"),
            "chunk_id": r["metadata"].get("chunk_id", r.get("id", "")),
            "score": r["score"],
        }
        for r in results
    ]
