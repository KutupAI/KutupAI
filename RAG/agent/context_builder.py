"""Build a bounded, traceable context window from retrieved legal chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

from RAG.vector_store.vector_store_interface import SearchResult
from RAG.configuration.rag_config_loader import agent_config


_NEXT_ARTICLE = re.compile(r"(?im)^\s*madde\s+(\d+)\b|\bmadde\s+(\d+)\s*[–-]")


@dataclass(frozen=True)
class ContextBuild:
    text: str
    sources: List[Dict[str, object]]
    estimated_tokens: int
    dropped_chunks: int


def _estimate_tokens(text: str) -> int:
    # Türkçe UTF-8 metin için bağımlılıksız ve temkinli token tahmini.
    return max(1, (len(text) + 3) // 4)


def _source_record(result: SearchResult, label: str) -> Dict[str, object]:
    meta = dict(result["metadata"])
    law_name = meta.get("law_name") or str(meta.get("source_file", "Bilinmeyen kaynak")).replace(".pdf", "").replace("_", " ")
    return {
        "label": label,
        # Katman sözleşmesinde kaynak pasajın kendisi de istenir. Bu alan,
        # kullanıcıya gösterilen kaynak ile LLM'e gönderilen kanıtın aynı
        # olmasını sağlar.
        "text": _article_scoped_text(result),
        "law_name": str(law_name),
        "law_number": str(meta.get("law_number") or "unknown"),
        "article_number": str(meta.get("article_no") or meta.get("article_number") or "unknown"),
        "source_file": str(meta.get("source_file") or meta.get("source") or "unknown"),
        "page_start": meta.get("page_start") or meta.get("page"),
        "page_end": meta.get("page_end") or meta.get("page"),
        "chunk_id": str(meta.get("chunk_id") or result["id"]),
        "document_id": str(meta.get("document_id") or ""),
        "score": float(result["score"]),
        "source_type": str(meta.get("source_type") or "unknown"),
        "document_category": str(meta.get("document_category") or ""),
        "authority_level": str(meta.get("authority_level") or "unknown"),
    }


def _article_scoped_text(result: SearchResult) -> str:
    """Do not leak the next article into a chunk labelled as the current one.

    PDF extraction can join the end of Madde 5 with the heading and first
    sentence of Madde 6.  A language model then reasonably but incorrectly
    treats both as the requested article.  Keep the text only until the first
    subsequent article heading whenever article metadata is available.
    """
    raw = str(result["text"] or "")
    current = str(result["metadata"].get("article_no") or result["metadata"].get("article_number") or "")
    if not current:
        return raw
    for match in _NEXT_ARTICLE.finditer(raw):
        article = match.group(1) or match.group(2)
        if article != current:
            return raw[: match.start()]
    return raw


def build_context(
    results: Iterable[SearchResult],
    *,
    max_context_tokens: int = agent_config.context_max_tokens,
    max_chunk_chars: int = agent_config.max_chunk_chars,
) -> ContextBuild:
    """Deduplicate, rank, label, and bound passages before generation.

    Every included passage receives a stable `[S<n>]` label.  The model may
    cite only these labels, which makes later citation validation deterministic.
    """
    ordered = sorted(results, key=lambda item: float(item["score"]), reverse=True)
    unique: List[SearchResult] = []
    seen: set[str] = set()
    for result in ordered:
        identity = " ".join(result["text"].split()).casefold()
        if identity and identity not in seen:
            seen.add(identity)
            unique.append(result)

    parts: List[str] = []
    sources: List[Dict[str, object]] = []
    used_tokens = 0
    for result in unique:
        label = f"S{len(sources) + 1}"
        source = _source_record(result, label)
        page = source["page_start"]
        page_text = f" | Sayfa {page}" if page else ""
        if source["source_type"] == "reference_docs":
            category = source["document_category"] or "belirsiz"
            header = (
                f"[{label}] Referans Belge | Kategori {category} | "
                f"Durum {source['authority_level']} | Dosya {source['source_file']}{page_text}"
            )
        elif source["source_type"] == "legal_facts":
            header = (
                f"[{label}] Yapılandırılmış Hukukî Olgu | Tür {result['metadata'].get('fact_type', 'unknown')} | "
                f"Dosya {source['source_file']}{page_text}"
            )
        else:
            header = (
                f"[{label}] {source['law_name']} | Kanun {source['law_number']} | "
                f"Madde {source['article_number']} | Dosya {source['source_file']}{page_text}"
            )
        body = " ".join(_article_scoped_text(result).split())[:max_chunk_chars].strip()
        candidate = f"{header}\n{body}"
        candidate_tokens = _estimate_tokens(candidate)
        if used_tokens + candidate_tokens > max_context_tokens:
            continue
        parts.append(candidate)
        sources.append(source)
        used_tokens += candidate_tokens

    return ContextBuild(
        text="\n\n".join(parts),
        sources=sources,
        estimated_tokens=used_tokens,
        dropped_chunks=max(0, len(unique) - len(sources)),
    )
