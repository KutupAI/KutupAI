"""Retrieval pipeline: expand → hybrid → PRF → CrossEncoder."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional, Set

from RAG.configuration.rag_config_loader import (
    prf_config,
    query_expansion_config,
    reranker_config,
    retrieval_config,
)
from RAG.retriever.hybrid import _rrf_fuse, hybrid_search
from RAG.retriever.prf import expand_query_with_prf
from RAG.retriever.query_expansion import apply_strategy
from RAG.retriever.reranker import rerank
from RAG.retriever.query_transform import transform_query
from RAG.retriever.source_policy import default_source_where
from RAG.vector_store.vector_store_interface import SearchResult, VectorStoreInterface


def _deduplicate(results: List[SearchResult]) -> List[SearchResult]:
    """Remove repeated passages before reranking without dropping distinct articles.

    The same legal text can occur in a full law and in a separately supplied
    extract.  Source-file-based de-duplication lets those copies consume two
    result slots, so the normalized text is the primary identity here.
    """
    unique: List[SearchResult] = []
    seen: Set[str] = set()
    for result in results:
        meta = result["metadata"]
        key = " ".join(result["text"].split()).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(result)
    return unique


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    *,
    mode: Optional[str] = None,
    use_prf: Optional[bool] = None,
    use_reranker: Optional[bool] = None,
    expansion_strategy: Optional[str] = None,
    use_graph: Optional[bool | str] = None,
    use_query_transform_llm: Optional[bool] = None,
    where: Optional[dict] = None,
    vector_store: Optional[VectorStoreInterface] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> List[SearchResult]:
    if not query or not str(query).strip():
        return []

    pipeline_started = perf_counter()

    k = min(max(top_k or retrieval_config.default_top_k, 1), retrieval_config.max_top_k)
    candidate_k = min(max(k * 3, retrieval_config.candidate_k), retrieval_config.max_candidate_k)
    where = default_source_where(query, where)

    strategy = expansion_strategy
    if strategy is None and query_expansion_config.enabled:
        strategy = query_expansion_config.selected_strategy
    # PRF ilk sonuçlardan sonra uygulanır; burada çalıştırmak aday listesi
    # oluşmadan aynı aramayı iki kez yapmak anlamına gelir.
    q = query.strip() if strategy == "prf" else (apply_strategy(query.strip(), strategy) if strategy else query.strip())

    started = perf_counter()
    transformed = transform_query(q, use_llm=use_query_transform_llm)
    if trace is not None:
        trace.update({
            "input_query": query.strip(), "retrieval_query": q, "top_k": k,
            "candidate_k": candidate_k, "mode": mode or retrieval_config.mode,
            "query_transform_ms": round((perf_counter() - started) * 1000, 3),
            "query_variants": transformed.queries, "query_transform_used_llm": transformed.used_llm,
        })
    search_started = perf_counter()
    search_traces: List[Dict[str, object]] = []
    variant_hits: List[List[SearchResult]] = []
    for variant in transformed.queries:
        variant_trace: Dict[str, object] = {"query": variant}
        variant_hits.append(
            hybrid_search(
                variant,
                top_k=candidate_k,
                where=where,
                mode=mode,
                vector_store=vector_store,
                metadata_query=q,
                trace=variant_trace,
            )
        )
        search_traces.append(variant_trace)
    if len(variant_hits) > 1:
        hits = _deduplicate(
            _rrf_fuse(
                variant_hits,
                # Özgün soru otoritedir; LLM yeniden yazımları yalnız recall'u
                # genişletir ve anlam kaydığında özgün soruyu geçemez.
                [1.0] + [0.65] * (len(variant_hits) - 1),
                retrieval_config.rrf_k,
                candidate_k,
            )
        )
    else:
        hits = _deduplicate(variant_hits[0] if variant_hits else [])
    if trace is not None:
        trace.update({
            "initial_search_ms": round((perf_counter() - search_started) * 1000, 3),
            "search_variants": search_traces,
            "initial_candidates_after_dedup": len(hits),
        })

    prf_started = perf_counter()
    prf_applied = False
    prf_query = None
    if (prf_config.enabled if use_prf is None else use_prf) and hits:
        prf_q = expand_query_with_prf(q, hits)
        if prf_q != q:
            prf_applied = True
            prf_query = prf_q
            prf_trace: Dict[str, object] = {"query": prf_q}
            hits = _deduplicate(
                hybrid_search(
                    prf_q,
                    top_k=candidate_k,
                    where=where,
                    mode=mode,
                    vector_store=vector_store,
                    metadata_query=q,
                    trace=prf_trace,
                )
            )
            q = prf_q
    if trace is not None:
        trace.update({
            "prf_enabled": bool(prf_config.enabled if use_prf is None else use_prf),
            "prf_applied": prf_applied,
            "prf_query": prf_query,
            "prf_ms": round((perf_counter() - prf_started) * 1000, 3),
            "candidates_after_prf": len(hits),
            "prf_search": prf_trace if prf_applied else None,
        })

    # Graph-RAG adayları reranker öncesinde tamamlar. Auto yalnız açık kanun+
    # madde sorularında çalışır; geniş sorularda anlam kaymasını önler.
    graph_enabled = use_graph is True or use_graph == "full"
    extracted = None
    if use_graph is None:
        from RAG.retriever.query_metadata import get_query_metadata_extractor

        extracted = get_query_metadata_extractor().extract(query)
        graph_enabled = bool(extracted.get("law_number") and extracted.get("article_no"))
        # Kesin atıfta metadata filtresi ilgili maddeyi zaten getirir. Komşuları
        # reranker öncesi eklemek doğru maddeyi düşürebilir; 200 soruluk test
        # bunu gösterdi. Bu yüzden Auto Graph-RAG ikinci ranker değil, eksik
        # madde için geri dönüş mekanizmasıdır.
        if graph_enabled:
            law = str(extracted["law_number"])
            article = str(extracted["article_no"])
            if any(
                str(item["metadata"].get("law_number") or "") == law
                and str(item["metadata"].get("article_no") or item["metadata"].get("article_number") or "") == article
                for item in hits
            ):
                graph_enabled = False
    graph_started = perf_counter()
    candidates_before_graph = len(hits)
    if graph_enabled and hits:
        from RAG.graph.legal_graph import get_legal_graph

        hits = _deduplicate(
            get_legal_graph().enrich(query, hits, include_references=(use_graph == "full"))
        )
    if trace is not None:
        trace.update({
            "graph_enabled": graph_enabled,
            "graph_ms": round((perf_counter() - graph_started) * 1000, 3),
            "candidates_before_graph": candidates_before_graph,
            "candidates_after_graph": len(hits),
        })

    reranker_started = perf_counter()
    if use_reranker is False:
        final = hits[:k]
    else:
        final = rerank(query=q, results=hits, top_k=k)
    if trace is not None:
        trace.update({
            "reranker_enabled": use_reranker is not False,
            "reranker_input_candidates": min(len(hits), max(k, reranker_config.top_n)),
            "reranker_ms": round((perf_counter() - reranker_started) * 1000, 3),
            "final_result_count": len(final),
            "total_retrieval_ms": round((perf_counter() - pipeline_started) * 1000, 3),
        })
    return final
