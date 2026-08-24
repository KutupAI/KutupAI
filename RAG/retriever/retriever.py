"""Retrieval pipeline: expand → hybrid → PRF → CrossEncoder."""

from __future__ import annotations

import re
from time import perf_counter
from typing import Any, Dict, List, Optional, Set

from RAG.configuration.rag_config_loader import (
    prf_config,
    query_expansion_config,
    reranker_config,
    retrieval_config,
    multi_hop_config,
    legal_index_config,
)
from RAG.retriever.hybrid import _rrf_fuse, hybrid_search
from RAG.retriever.prf import expand_query_with_prf
from RAG.retriever.query_expansion import apply_strategy
from RAG.retriever.reranker import rerank
from RAG.retriever.query_transform import transform_query
from RAG.retriever.query_frame import EvidenceSlot, QueryFrame, build_query_frame
from RAG.retriever.query_metadata import get_query_metadata_extractor
from RAG.retriever.source_policy import default_source_where
from RAG.retriever.text_utils import fold_turkish
from RAG.vector_store.vector_store_interface import SearchResult, VectorStoreInterface


def _referenced_law_title_queries(query: str) -> List[str]:
    """Kabul tarihi/kanun no istenen, numarası yazılmamış kanun adlarını ayırır."""
    normalized = fold_turkish(query).casefold()
    if not any(term in normalized for term in ("kabul tarihi", "kanun numarasi", "yasa numarasi")):
        return []
    match = re.search(r"\bkanunu(?:['’]?[a-zıiüöşç]+)?\s+(?:kabul tarihi|kanun numarasi|yasa numarasi)", normalized)
    if not match:
        return []
    before = query[:match.start()]
    words = re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿĞğİıŞşÇçÜüÖö]+", before)
    # Son iki-dört sözcükten aday üretmek, "... olduğu Su Ürünleri Kanunu"
    # gibi doğal cümlelerde başlığın önündeki bağlaçları taşımadan çalışır.
    titles: List[str] = []
    for width in (2, 3, 4):
        if len(words) >= width:
            titles.append(" ".join([*words[-width:], "Kanunu"]))
    return list(dict.fromkeys(titles))


def _multi_hop_queries(query: str) -> List[str]:
    if not multi_hop_config.enabled:
        return []
    parts = re.split(r"\b(?:ayrıca|bunun yanında|diğer yandan|ek olarak)\b|;", query, flags=re.IGNORECASE)
    if len(parts) < 2:
        return []
    return [" ".join(part.split()) for part in parts if len(part.split()) >= 5][: multi_hop_config.max_subqueries]


def _multi_law_candidates(query: str, frame: QueryFrame, candidate_k: int, mode: Optional[str]) -> List[SearchResult]:
    """Her açık kanun için ayrı aday havuzu getirir."""
    if len(frame.law_numbers) < 2:
        return []
    combined: List[SearchResult] = []
    for law_number in frame.law_numbers[:3]:
        candidates = hybrid_search(
            query, top_k=candidate_k, mode=mode,
            where={"law_number": law_number, "source_type": "laws"}, metadata_query=query,
        )
        for candidate in candidates:
            candidate["metadata"] = dict(candidate["metadata"])
            candidate["metadata"].setdefault("evidence_slot", f"law:{law_number}")
        combined.extend(candidates)
    return _deduplicate(combined)


def _soft_law_candidates(query: str, frame: QueryFrame, candidate_k: int, mode: Optional[str]) -> List[SearchResult]:
    """Belirsiz kanun adı için geniş aramaya ek, daraltmayan aday havuzu.

    Örneğin "askerî disiplin" birden çok mevzuata işaret edebilir. Bu durumda
    tahmin edilen kanunu tek sonuç alanı yapmak yerine, genel hybrid sonuca
    ikinci bir sinyal olarak ekleriz. Reranker iki havuzu birlikte değerlendirir.
    """
    primary = str(frame.intent.primary_law_number or "")
    if not primary or primary in frame.strict_law_numbers:
        return []
    candidates = hybrid_search(
        query,
        top_k=min(candidate_k, 12),
        mode=mode,
        where={"law_number": primary, "source_type": "laws"},
        metadata_query=query,
    )
    for candidate in candidates:
        candidate["metadata"] = dict(candidate["metadata"])
        candidate["metadata"]["retrieval_scope"] = "soft_law_hint"
    return candidates


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


def _requested_source_types(where: Optional[dict]) -> set[str]:
    """Return an explicit multi-corpus scope, if the caller requested one."""
    source_type = (where or {}).get("source_type")
    if isinstance(source_type, dict) and isinstance(source_type.get("$in"), list):
        return {str(value) for value in source_type["$in"]}
    return set()


def _ensure_source_coverage(
    final: List[SearchResult],
    candidates: List[SearchResult],
    *,
    required_types: set[str],
    top_k: int,
) -> List[SearchResult]:
    """Keep evidence from each requested corpus when it is actually available.

    Mixed questions can ask both for a legal basis and a document example. A
    pure score sort may fill every final slot with laws even when a relevant
    form/template was retrieved. This tiny diversification runs only for an
    explicit ``source_type: {$in: [...]}`` scope; it never changes ordinary
    legal-only retrieval.
    """
    if len(required_types) < 2:
        return final[:top_k]

    selected = list(final[:top_k])
    selected_ids = {str(item["id"]) for item in selected}
    present = {str(item["metadata"].get("source_type") or "") for item in selected}

    for source_type in sorted(required_types - present):
        replacement = next(
            (
                item for item in candidates
                if str(item["metadata"].get("source_type") or "") == source_type
                and str(item["id"]) not in selected_ids
            ),
            None,
        )
        if replacement is None:
            continue
        # En düşük öncelikli, zaten birden fazla temsil edilen corpus sonucunu
        # çıkar. Böylece kaynak kapsaması korunurken top_k sabit kalır.
        removable_index = next(
            (
                index for index in range(len(selected) - 1, -1, -1)
                if str(selected[index]["metadata"].get("source_type") or "") in present
                and sum(
                    str(item["metadata"].get("source_type") or "")
                    == str(selected[index]["metadata"].get("source_type") or "")
                    for item in selected
                ) > 1
            ),
            None,
        )
        if removable_index is not None:
            removed = selected.pop(removable_index)
            selected_ids.discard(str(removed["id"]))
        elif len(selected) >= top_k:
            selected.pop()
        selected.append(replacement)
        selected_ids.add(str(replacement["id"]))
        present.add(source_type)

    return selected[:top_k]


def _matches_slot(result: SearchResult, slot: EvidenceSlot) -> bool:
    metadata = result["metadata"]
    law = str(metadata.get("law_number") or "")
    article = str(metadata.get("article_no") or metadata.get("article_number") or "")
    marker = str(metadata.get("evidence_slot") or "")
    if slot.kind == "law":
        return law == str(slot.law_number or "")
    if slot.kind == "article":
        return law == str(slot.law_number or law) and article == str(slot.article_no or "")
    if slot.kind == "amendment":
        if marker != "amendment" and str(metadata.get("source_type") or "") != "amendment_ledger":
            return False
        if slot.law_number and law != str(slot.law_number):
            return False
        candidate_number = str(metadata.get("amending_number") or "")
        return not slot.amending_number or candidate_number == str(slot.amending_number)
    if slot.kind == "duration":
        # Süre slotu için ham madde metadata'sında özel bir işaret yoktur.
        # Yalnız açık süre ifadesi olan pasajlar, çok-kanıtlı soruda tamamlayıcı
        # aday olabilir; sıradan tarih cetvelleri bu slotu karşılamaz.
        return bool(re.search(r"\b\d+\s*(?:saat|gün|gun|ay|yıl|yil)\b|\bbir\s+hafta\b", result["text"], re.IGNORECASE))
    return marker == slot.kind or str(metadata.get("structured_fact_type") or "") in {
        "constitutional_court_annulment" if slot.kind == "court" else "legal_time_limit"
    }


def _ensure_slot_coverage(
    final: List[SearchResult], candidates: List[SearchResult], *, frame: QueryFrame, top_k: int
) -> List[SearchResult]:
    """Çok-hükümlü sorularda tek kanunun bütün Top-K'yi doldurmasını önler."""
    if not frame.needs_multiple_evidence or not frame.slots:
        return final[:top_k]
    selected = list(final[:top_k])
    selected_ids = {str(item["id"]) for item in selected}
    for slot in frame.slots:
        if any(_matches_slot(item, slot) for item in selected):
            continue
        candidate = next(
            (item for item in candidates if str(item["id"]) not in selected_ids and _matches_slot(item, slot)),
            None,
        )
        if candidate is None:
            continue
        if len(selected) >= top_k:
            selected.pop()
        selected.append(candidate)
        selected_ids.add(str(candidate["id"]))
    return selected[:top_k]


def _ensure_structured_evidence(
    final: List[SearchResult], structured: List[SearchResult], *, frame: QueryFrame, top_k: int
) -> List[SearchResult]:
    """Açık tablo/mahkeme kanıtının raw chunk'lar arasında kaybolmasını önler."""
    if not structured or not (frame.needs_amendment_evidence or any(slot.kind == "court" for slot in frame.slots)):
        return final[:top_k]
    selected = list(final[:top_k])
    selected_ids = {str(item["id"]) for item in selected}
    # Bir karşılaştırma iki (hatta daha çok) amendment/court kanıtı ister.
    # Eski yol ilk structured kaydı ekleyip diğer gerekli satırları atıyordu.
    required_slots = [slot for slot in frame.slots if slot.kind in {"amendment", "court"}]
    if not required_slots:
        required_slots = [None]
    for slot in required_slots:
        if slot is not None and any(_matches_slot(item, slot) for item in selected):
            continue
        required = next(
            (item for item in structured if str(item["id"]) not in selected_ids and (slot is None or _matches_slot(item, slot))),
            None,
        )
        if required is None:
            continue
        if len(selected) >= top_k:
            # Zorunlu kanıt eklenirken daha önce eklenmiş başka zorunlu kanıtı
            # çıkarmamaya çalışır.
            removable = next(
                (index for index in range(len(selected) - 1, -1, -1)
                 if not any(_matches_slot(selected[index], required_slot) for required_slot in required_slots if required_slot is not None)),
                len(selected) - 1,
            )
            selected.pop(removable)
        selected.insert(0, required)
        selected_ids.add(str(required["id"]))
    return selected[:top_k]


def _ensure_reference_title_evidence(
    final: List[SearchResult], reference_hits: List[SearchResult], *, top_k: int
) -> List[SearchResult]:
    """Numarası sorulan atıf kanıtının reranker sonrası da kalmasını sağlar."""
    if not reference_hits:
        return final[:top_k]
    selected = list(final[:top_k])
    selected_ids = {str(item["id"]) for item in selected}
    if any(str(item["id"]) in {str(hit["id"]) for hit in reference_hits} for item in selected):
        return selected
    evidence = next((item for item in reference_hits if str(item["id"]) not in selected_ids), None)
    if evidence is None:
        return selected
    # Değişiklik cetveli gibi yapılandırılmış zorunlu kanıtı çıkarmadan, en
    # düşük öncelikli ham maddeyi atıf kanıtıyla değiştirir.
    removable = next(
        (
            index for index in range(len(selected) - 1, -1, -1)
            if str(selected[index]["metadata"].get("source_type") or "") != "amendment_ledger"
        ),
        len(selected) - 1,
    )
    if selected:
        selected.pop(removable)
    selected.append(evidence)
    return selected[:top_k]


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
    frame = build_query_frame(query)

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
    variants = list(dict.fromkeys(transformed.queries + _multi_hop_queries(q)))
    search_started = perf_counter()
    search_traces: List[Dict[str, object]] = []
    variant_hits: List[List[SearchResult]] = []
    for variant in variants:
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
    intent = frame.intent
    law_scoped_hits = _multi_law_candidates(query, frame, candidate_k, mode)
    soft_law_hits = _soft_law_candidates(query, frame, candidate_k, mode)
    structured_hits: List[SearchResult] = []
    fts_hits: List[SearchResult] = []
    reference_title_hits: List[SearchResult] = []
    legal_index = None
    try:
        from RAG.metadata.legal_index import get_legal_index

        legal_index = get_legal_index()
        structured_hits = legal_index.structured_lookup(q, frame, top_k=candidate_k)
        for title_query in _referenced_law_title_queries(query):
            title_hits = legal_index.fts_all_terms_search(title_query, top_k=20)
            # Başka kanun adına ait metadata çoğu zaman sorudaki ana kanunun
            # görev/atıf maddesinde yazılıdır. Ana kanundan sonuç varsa onu
            # tercih etmek, "ürünleri" gibi genel sözcüklerin alakasız FTS
            # eşleşmelerini kanıt penceresine sokmaz.
            primary_law = str(frame.intent.primary_law_number or "")
            primary_hits = [
                item for item in title_hits
                if primary_law and str(item["metadata"].get("law_number") or "") == primary_law
            ]
            reference_title_hits.extend(primary_hits or title_hits[:4])
        # Hybrid zaten BM25 taşır. Ayrı FTS tüm serbest sorulara zorla katılırsa
        # ilk 15 reranker adayını gereksiz tam-terim parçalarıyla doldurur.
        # Bu nedenle FTS sadece ifade/terim ağırlıklı hukukî soru türlerinde ek adaydır.
        if frame.kind in {"authority", "sanction", "condition", "comparison", "temporal", "amendment", "multi_law_relation", "general"}:
            fts_scope = frame.law_numbers if len(frame.law_numbers) == 1 else ()
            # Comparison/temporal sorularda doğru madde FTS'te ilk 20'nin
            # hemen altında kalabiliyor. Bu yalnız aday havuzunu büyütür;
            # nihai seçim yine cross-encoder reranker tarafından yapılır.
            fts_limit = max(
                legal_index_config.fts_candidate_k,
                candidate_k * (2 if frame.kind in {"comparison", "temporal"} else 1),
            )
            fts_hits = legal_index.fts_search(q, top_k=fts_limit, law_numbers=fts_scope)
    except Exception:
        legal_index = None
    reference_title_hits = _deduplicate(reference_title_hits)
    supplemental_hits = _deduplicate(law_scoped_hits + soft_law_hits + fts_hits)
    if supplemental_hits:
        # Ek havuzlar RRF ile birleşir; öne eklenip normal semantic adayları
        # reranker'ın ilk penceresinden çıkarmaz.
        hits = _deduplicate(
            _rrf_fuse(
                [hits, supplemental_hits], [1.0, 0.60], retrieval_config.rrf_k, candidate_k
            )
        )
    # Çok parçalı soruda atıf yapılan kanunun kimlik bilgisi ayrı bir kanıttır.
    # Bu dar başlık araması reranker penceresine girer; amendment ledger ise
    # aşağıdaki structured-coverage adımıyla ayrıca korunur.
    if reference_title_hits:
        hits = _deduplicate(reference_title_hits + hits)
    if trace is not None:
        trace.update({
            "initial_search_ms": round((perf_counter() - search_started) * 1000, 3),
            "search_variants": search_traces,
            "multi_hop_queries": variants[len(transformed.queries) :],
            "initial_candidates_after_dedup": len(hits),
            "multi_law_candidates": len(law_scoped_hits),
            "soft_law_candidates": len(soft_law_hits),
            "structured_candidates": len(structured_hits),
            "fts_candidates": len(fts_hits),
            "reference_title_candidates": len(reference_title_hits),
            "evidence_slots": [slot.key for slot in frame.slots],
            "query_intent": intent.kind,
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
            # PRF ile gelen yeni adaylar structured/tablo kanıtlarını silmez;
            # aynı ham chunk gelirse structured metadata sürümü korunur.
            hits = _deduplicate(
                _rrf_fuse(
                    [hits, _deduplicate(law_scoped_hits + soft_law_hits + fts_hits)],
                    [1.0, 0.60], retrieval_config.rrf_k, candidate_k,
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
    # Graph-RAG yalnız açık kanunlar arası bağ/atıf sorusunda genişletilir.
    # Değişiklik cetveli ve karşılaştırma soruları zaten kendi kanıt yoluna
    # sahiptir; graph komşularının eklenmesi bu sorularda gürültü üretir.
    cross_law_cues = ("atif", "atıf", "diger kanun", "başka kanun", "hangi kanun", "baglanti", "iliski")
    normalized_query = query.casefold()
    if use_graph == "full" and (
        frame.needs_amendment_evidence
        or frame.kind == "comparison"
        # İki kanun numarası açıkça soruda geçiyorsa ikinci numara değiştirici
        # düzenleme olsa bile Graph-RAG onu ilişki adayı olarak değerlendirebilir.
        # Kesin filtre yalnız ilk hedef kanunu daraltır; graph için iki açık
        # referans yeterlidir.
        or (len(frame.law_numbers) < 2 and not any(cue in normalized_query for cue in cross_law_cues))
    ):
        graph_enabled = False
    extracted = None
    if use_graph is None:
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
    graph_law_candidates = 0
    graph_referenced_laws: List[str] = []
    if graph_enabled and hits:
        from RAG.graph.legal_graph import get_legal_graph

        graph = get_legal_graph()
        if use_graph == "full" and intent.kind == "multi_law_relation":
            graph_referenced_laws = graph.referenced_laws(hits)
            for law_number in graph_referenced_laws:
                extra = hybrid_search(
                    query, top_k=max(k, candidate_k // 2), mode=mode,
                    where={"law_number": law_number, "source_type": "laws"}, metadata_query=query,
                )
                graph_law_candidates += len(extra)
                hits = _deduplicate(hits + extra)
        hits = _deduplicate(graph.enrich(query, hits, include_references=(use_graph == "full")))
    if trace is not None:
        trace.update({
            "graph_enabled": graph_enabled,
            "graph_ms": round((perf_counter() - graph_started) * 1000, 3),
            "candidates_before_graph": candidates_before_graph,
            "candidates_after_graph": len(hits),
            "graph_referenced_laws": graph_referenced_laws,
            "graph_law_candidates": graph_law_candidates,
        })

    reranker_started = perf_counter()
    if use_reranker is False:
        final = hits[:k]
    else:
        final = rerank(query=q, results=hits, top_k=k)

    # JSON ledger/fact yolu, SQLite yoksa tam geri dönüş; SQLite varsa yalnız
    # structured indeksin açık bir amendment slotunu bulamadığı durumda güvenli
    # tamamlayıcıdır. Böylece eski, doğrulanmış cetvel recall'u kaybolmaz.
    ledger_hits: List[SearchResult] = []
    fact_hits: List[SearchResult] = []
    structured_amendment_found = any(
        str(item["metadata"].get("source_type") or "") == "amendment_ledger"
        for item in structured_hits
    )
    if legal_index is None or not legal_index.available() or (
        frame.needs_amendment_evidence and not structured_amendment_found
    ):
        from RAG.retriever.amendment_ledger import lookup_amendment_ledger
        from RAG.retriever.fact_lookup import lookup_facts

        target_law = (
            frame.intent.primary_law_number
            if frame.intent.primary_law_number in frame.strict_law_numbers
            else None
        )
        ledger_hits = lookup_amendment_ledger(query, str(target_law) if target_law else None)
        # Fact JSON araması yalnız eski SQLite-yok yolunda çalışır; aksi hâlde
        # farklı kanunlardan "iptal" kayıtları final sonuçlarını kirletebilir.
        if legal_index is None or not legal_index.available():
            evidence_laws = {str(item["metadata"].get("law_number") or "unknown") for item in final}
            fact_hits = lookup_facts(query, allowed_law_numbers=evidence_laws)
        final = _deduplicate(ledger_hits + final + fact_hits)
    final = _ensure_structured_evidence(final, structured_hits, frame=frame, top_k=k)
    final = _ensure_reference_title_evidence(final, reference_title_hits, top_k=k)
    final = _ensure_slot_coverage(final, hits, frame=frame, top_k=k)
    requested_source_types = _requested_source_types(where)
    final = _ensure_source_coverage(
        final,
        hits,
        required_types=requested_source_types,
        top_k=k,
    )
    if trace is not None:
        trace.update({
            "reranker_enabled": use_reranker is not False,
            "reranker_input_candidates": min(len(hits), max(k, reranker_config.top_n)),
            "reranker_ms": round((perf_counter() - reranker_started) * 1000, 3),
            "final_result_count": len(final),
            "amendment_ledger_results": len(ledger_hits),
            "facts_registry_results": len(fact_hits),
            "structured_evidence_results": len(structured_hits),
            "source_coverage_types": sorted(requested_source_types),
            "final_source_types": sorted({str(item["metadata"].get("source_type") or "") for item in final}),
            "total_retrieval_ms": round((perf_counter() - pipeline_started) * 1000, 3),
        })
    return final
