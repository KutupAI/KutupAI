"""
Hibrit Geri Getirme (Hybrid Retrieval): BM25 + Vektör Araması + RRF Füzyonu
---------------------------------------------------------------------------
Sorgudan otomatik olarak meta veri filtrelerini (Classification Agent) çıkarır ve 
arama uzayını daraltarak sadece ilgili kanun ve maddede arama yapar.

🚀 Geliştirilmiş Versiyon:
- Logging entegrasyonu eklendi.
- Otomatik filtreleme hatası durumunda güvenli geri dönüş (fallback) mekanizması eklendi.
- Kod yapısı modüler hale getirildi (_vector_search fonksiyonu).
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Dict, List, Optional

from RAG.configuration.rag_config_loader import retrieval_config
from RAG.retriever.bm25_index import get_bm25_index
from RAG.retriever.query_metadata import get_query_metadata_extractor
from RAG.vector_store.chroma_store import get_vector_store
from RAG.vector_store.vector_store_interface import SearchResult, VectorStoreInterface

# Logger tanımlama
logger = logging.getLogger(__name__)


def _rrf_fuse(
    lists: List[List[SearchResult]],
    weights: List[float],
    rrf_k: int,
    top_k: int,
) -> List[SearchResult]:
    """Reciprocal Rank Fusion (RRF) algoritması ile sonuçları birleştirir."""
    scores: Dict[str, float] = {}
    payload: Dict[str, SearchResult] = {}
    
    for results, weight in zip(lists, weights):
        for rank, item in enumerate(results, start=1):
            key = item["id"] or str(hash(item["text"]))
            # RRF Skoru Hesaplama: 1 / (k + rank)
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (rrf_k + rank))
            
            # En yüksek skora sahip ögeyi sakla (metadata için gerekli olabilir)
            if key not in payload or item["score"] > payload[key]["score"]:
                payload[key] = item

    # Skorlara göre sıralama ve ilk top_k elemanı alma
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    
    return [
        SearchResult(
            id=payload[key]["id"],
            text=payload[key]["text"],
            metadata={**payload[key]["metadata"], "rrf_score": round(score, 6)},
            score=round(score, 6),
        )
        for key, score in ordered
    ]


def _filter(results: List[SearchResult], where: Optional[dict]) -> List[SearchResult]:
    """BM25 sonuçlarını meta verisine göre filtreler."""
    if not where:
        return results
    
    # Basit eşleşme kontrolü
    filtered_results = []
    for r in results:
        match = True
        for k, v in where.items():
            actual = r["metadata"].get(k)
            if isinstance(v, dict) and "$in" in v:
                if actual not in v["$in"]:
                    match = False
                    break
            elif actual != v:
                match = False
                break
        if match:
            filtered_results.append(r)
            
    return filtered_results


def _to_chroma_where(where: Optional[dict]) -> Optional[dict]:
    """Translate a flat public filter into Chroma's multi-field syntax."""
    if not where:
        return None
    return {"$and": [{key: value} for key, value in where.items()]} if len(where) > 1 else where


def _vector_search(
    store: VectorStoreInterface,
    query: str,
    top_k: int,
    *,
    where: Optional[dict],
    auto_filter: bool,
    fallback_where: Optional[dict] = None,
) -> List[SearchResult]:
    """
    ChromaDB'den vektörel arama yapar.
    
    🚀 Güvenlik Mekanizması:
    Eğer 'auto_filter' True ise (yani filtre sorgudan otomatik çıkarıldıysa) ve 
    ChromaDB hata verirsa (örn: alan eksik veya tip uyuşmazlığı), 
    filtre olmadan tekrar arama yaparak sistemi çökmeden kurtarır.
    
    Args:
        query: Arama sorgusu.
        top_k: İstenen sonuç sayısı.
        where: Filtreleme koşulları.
        auto_filter: Filtrenin otomatik mi yoksa kullanıcı tarafından mı geldiğini belirtir.
        
    Returns:
        SearchResults listesi.
    """
    try:
        # Normal arama denemesi
        results = store.similarity_search(query, top_k, where=where)
        return results
        
    except Exception as e:
        # Hata yakalandı
        if not auto_filter or not where:
            # Eğer filtre manuel geldiyse veya filtre yoksa, hatayı fırlat (kritik hata)
            logger.error(f"Vector search failed with explicit/no filter: {e}", exc_info=True)
            raise
            
        # Çıkarılan metadata filtresi geçersiz olsa da çağıranın filtresi korunur.
        # Filtresiz geri dönüş, kaynak veya kiracı kapsamını aşabilir.
        logger.warning(
            f"Automatic metadata filter failed ({e}). Retrying unfiltered search to ensure stability.",
            exc_info=True
        )
        try:
            return store.similarity_search(query, top_k, where=fallback_where)
        except Exception as fallback_error:
            # Fallback de başarısız olduysa, boş liste döndür veya hatayı fırlat
            logger.critical("Unfiltered vector search also failed.", exc_info=True)
            return []


def hybrid_search(
    query: str,
    top_k: int,
    *,
    where: Optional[dict] = None,
    mode: Optional[str] = None,
    vector_store: Optional[VectorStoreInterface] = None,
    metadata_query: Optional[str] = None,
    trace: Optional[Dict[str, object]] = None,
) -> List[SearchResult]:
    """
    Ana hibrit arama fonksiyonu.
    
    Modlar:
    - 'vector': Sadece vektörel arama.
    - 'bm25': Sadece kelime bazlı arama.
    - 'hybrid': Her ikisini RRF ile birleştirir (Varsayılan).
    """
    mode = (mode or retrieval_config.mode).lower()
    # Aday havuzunu biraz daha geniş tutuyoruz ki RRF iyi çalışabilsin
    k = max(top_k, retrieval_config.candidate_k)

    # 🚀 CLASSIFICATION AGENT MANTIĞI: Filtre sağlanmadıysa sorgudan otomatik çıkar
    explicit_where = dict(where or {})
    auto_filter = False
    extractor = get_query_metadata_extractor()
    # Query transform arama ifadesini çeşitlendirebilir; ancak kesin kanun/madde
    # filtresini yalnız kullanıcının özgün sorusu belirler.
    extracted_filters = extractor.extract(metadata_query or query)
    if extracted_filters:
        # Açık çağıran filtresi önceliklidir; çıkarılan kanun/madde filtresi eklenir.
        # Böylece kaynak kapsamı ve kesin kanun araması birlikte çalışır.
        where = {**extracted_filters, **explicit_where}
        auto_filter = True
        logger.debug("Automatic metadata filter applied: %s", where)
    else:
        where = explicit_where or None
    if trace is not None:
        trace.update({"mode": mode, "metadata_filter": where or {}, "auto_filter": auto_filter})

    # 🚀 CHROMADB FIX: Birden fazla filtre varsa $and operatörü ile sarmala
    # ChromaDB tek bir dict içinde birden fazla key kabul etmez, $and gerektirir.
    store = vector_store or get_vector_store()
    is_chroma = store.__class__.__name__ == "ChromaStore"
    chroma_where = _to_chroma_where(where) if is_chroma else where
    fallback_chroma_where = _to_chroma_where(explicit_where) if is_chroma else (explicit_where or None)

    # --- ARAMA MODLARINA GÖRE DALMA ---

    if mode == "vector":
        # Sadece Vektörel Arama
        started = perf_counter()
        results = _vector_search(store, query, k, where=chroma_where, auto_filter=auto_filter, fallback_where=fallback_chroma_where)
        if trace is not None:
            trace.update({"vector_ms": round((perf_counter() - started) * 1000, 3), "vector_candidates": len(results), "result_count": min(len(results), top_k)})
        return results[:top_k]

    if mode == "bm25":
        # Sadece BM25 Arama
        # BM25 kendi _filter fonksiyonumuzu kullanır, o düz dict kabul eder
        started = perf_counter()
        results = get_bm25_index().search(query, k, where=where)
        if trace is not None:
            trace.update({"bm25_ms": round((perf_counter() - started) * 1000, 3), "bm25_candidates": len(results), "result_count": min(len(results), top_k)})
        return results[:top_k]

    # --- HİBRİT MOD (VARSAYILAN) ---
    
    # 1. Vektörel Arama (Güvenli Wrapper ile)
    started = perf_counter()
    vec_results = _vector_search(store, query, k, where=chroma_where, auto_filter=auto_filter, fallback_where=fallback_chroma_where)
    vector_ms = (perf_counter() - started) * 1000
    
    # 2. BM25 Arama
    started = perf_counter()
    lex_results = get_bm25_index().search(query, k, where=where)
    bm25_ms = (perf_counter() - started) * 1000
    
    # 3. RRF ile Birleştirme
    started = perf_counter()
    fused_results = _rrf_fuse(
        [vec_results, lex_results],
        [retrieval_config.vector_weight, retrieval_config.bm25_weight],
        retrieval_config.rrf_k,
        top_k,
    )
    if trace is not None:
        trace.update({
            "vector_ms": round(vector_ms, 3), "vector_candidates": len(vec_results),
            "bm25_ms": round(bm25_ms, 3), "bm25_candidates": len(lex_results),
            "rrf_ms": round((perf_counter() - started) * 1000, 3),
            "fused_candidates": len(fused_results), "result_count": len(fused_results),
        })
    
    return fused_results
