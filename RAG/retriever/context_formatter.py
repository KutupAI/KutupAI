"""
Geri Getirilen Sonuçları Ajanlar İçin Biçimlendirme (Context + Sources)
-----------------------------------------------------------------------
LLM'e gönderilecek bağlam metnini ve kaynak listesini hazırlar.
KutupAI metadata yapısına (article_no / law_number) tam uyumludur.
"""

from __future__ import annotations

from typing import Any, Dict, List

from RAG.vector_store.vector_store_interface import SearchResult


def format_context(results: List[SearchResult]) -> str:
    """Sonuçları LLM'in anlayacağı yapılandırılmış bir metne dönüştürür."""
    if not results:
        return "İlgili herhangi bir hukuki pasaj bulunamadı."
    
    parts = []
    for i, r in enumerate(results, start=1):
        m = r["metadata"]
        
        art_no = m.get('article_no') or m.get('article_number', '?')
        
        law_name = m.get('law_name')
        if not law_name or law_name == 'unknown':
            source_file = m.get('source_file') or m.get('source', '')
            law_name = source_file.replace('.pdf', '').replace('_', ' ') if source_file else 'Bilinmeyen Kanun'
            
        source_file = m.get('source_file') or m.get('source', '?')
        page_start = m.get("page_start") or m.get("page")
        page_end = m.get("page_end") or m.get("page")
        page_label = ""
        if page_start:
            page_label = f" | Sayfa: {page_start}" if page_start == page_end else f" | Sayfalar: {page_start}-{page_end}"
        
        header = (
            f"[KAYNAK {i}] {law_name} | Madde {art_no} | Dosya: {source_file}{page_label} | Skor: {r['score']:.4f}"
        )
        parts.append(f"{header}\n{r['text']}")
        
    return "\n\n".join(parts)


def extract_sources(results: List[SearchResult]) -> List[Dict[str, Any]]:
    """Kaynakça listesini sözlük formatında döndürür."""
    sources = []
    for r in results:
        m = r["metadata"]
        sources.append({
            "law_name": m.get("law_name", "unknown"),
            "article_no": m.get("article_no") or m.get("article_number", "unknown"),
            "law_number": m.get("law_number", "unknown"),
            "source_file": m.get("source_file") or m.get("source", "unknown"),
            "source_type": m.get("source_type", "unknown"),
            "page_start": m.get("page_start") or m.get("page"),
            "page_end": m.get("page_end") or m.get("page"),
            "chunk_id": m.get("chunk_id", r.get("id", "")),
            "score": r["score"],
        })
    return sources
