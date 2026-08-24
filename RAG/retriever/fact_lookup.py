"""Yapılandırılmış olguları kanıt pasajlarıyla birlikte retrieval sonucuna ekler."""

from __future__ import annotations

import json
import re
from pathlib import Path

from RAG.configuration.rag_config_loader import facts_registry_config
from RAG.retriever.text_utils import fold_turkish, tokenize
from RAG.vector_store.vector_store_interface import SearchResult


_IDENTIFIER = re.compile(r"\b(?:khk\s*[-/]?\s*)?(\d{3,5})(?:/\d+)?\b", re.IGNORECASE)
_STOP = {"kanun", "kanunu", "madde", "birinci", "hangi", "nedir", "karar", "tarihli", "sayili", "hukum"}


def _text(record: dict) -> str:
    values = record.get("values") or {}
    evidence = record.get("evidence") or {}
    return " ".join([
        str(record.get("fact_type") or ""), json.dumps(values, ensure_ascii=False),
        str(record.get("evidence_text") or ""), json.dumps(evidence, ensure_ascii=False),
    ])


def lookup_facts(query: str, *, allowed_law_numbers: set[str] | None = None) -> list[SearchResult]:
    """Fact aramasını ana chunk retrieval'ın doğruladığı kanunlarla sınırlar."""
    if not facts_registry_config.enabled or not query.strip():
        return []
    path: Path = facts_registry_config.output_path
    if not path.is_file():
        return []
    normalized = fold_turkish(query).casefold()
    identifiers = set(_IDENTIFIER.findall(normalized))
    terms = {term for term in tokenize(normalized, min_len=5) if term not in _STOP}
    records = json.loads(path.read_text(encoding="utf-8")).get("records", [])
    semantic_hits: list[SearchResult] = []
    try:
        from RAG.vector_store.fact_store import get_fact_store

        semantic_hits = get_fact_store().search(query, facts_registry_config.max_results * 50)
    except Exception:
        # Fact indeksi henüz oluşturulmadıysa temel chunk retrieval kesintisiz devam eder.
        semantic_hits = []
    allowed = {str(value) for value in (allowed_law_numbers or set()) if value and str(value) != "unknown"}
    if allowed:
        semantic_hits = [
            item for item in semantic_hits
            if str(item["metadata"].get("law_number") or "unknown") in allowed
        ]
    if not identifiers:
        return semantic_hits[: facts_registry_config.max_results]
    ranked: list[tuple[int, dict]] = []
    for record in records:
        record_law = str((record.get("evidence") or {}).get("law_number") or "unknown")
        if allowed and record_law not in allowed:
            continue
        haystack = fold_turkish(_text(record)).casefold()
        matched_ids = sum(bool(re.search(rf"\b{re.escape(value)}\b", haystack)) for value in identifiers)
        if not matched_ids:
            continue
        matched_terms = sum(term in haystack for term in terms)
        ranked.append((matched_ids * 20 + matched_terms, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    lexical_hits: list[SearchResult] = []
    for score, record in ranked[: facts_registry_config.max_results]:
        evidence = dict(record.get("evidence") or {})
        values = dict(record.get("values") or {})
        fact_type = str(record.get("fact_type") or "legal_fact")
        text = f"Yapılandırılmış hukukî olgu ({fact_type}): {json.dumps(values, ensure_ascii=False)}. Kanıt: {record.get('evidence_text', '')}"
        lexical_hits.append(SearchResult(
            id=f"fact:{record.get('fact_id')}", text=text,
            metadata={
                **evidence,
                "chunk_id": f"fact:{record.get('fact_id')}",
                "law_name": f"Yapılandırılmış hukukî olgu: {fact_type}",
                "source_type": "legal_facts",
                "fact_type": fact_type,
                "fact_values": json.dumps(values, ensure_ascii=False),
            },
            score=round(min(0.995, 0.84 + score / 1000), 4),
        ))
    combined: dict[str, SearchResult] = {item["id"]: item for item in semantic_hits}
    for item in lexical_hits:
        combined[item["id"]] = item
    return sorted(combined.values(), key=lambda item: float(item["score"]), reverse=True)[: facts_registry_config.max_results]
