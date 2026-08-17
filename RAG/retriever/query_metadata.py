"""
Sorgu Meta Veri Çıkarıcı (Classification Agent Mantığı) - Geliştirilmiş Versiyon
-------------------------------------------------------
Kullanıcı sorgularından kanun numarası (law_number) ve madde numarasını (article_no) 
çıkararak ChromaDB ve BM25 üzerinde sıkı ön filtreleme (strict pre-filtering) yapılmasını sağlar.
"""

import re
from functools import lru_cache
from typing import Dict, Any

from RAG.retriever.text_utils import fold_turkish, tokenize


def _title_tokens(value: str) -> set[str]:
    """Meaningful tokens in a source title; numbers and generic words vanish."""
    generic = {"kanunu", "yonetmeligi", "yonetmelik", "tebligi", "teblig", "tuzugu", "tuzuk", "sayili"}
    return {token for token in tokenize(fold_turkish(value).casefold(), min_len=3) if not token.isdigit() and token not in generic}


@lru_cache(maxsize=512)
def _law_from_index(normalized_query: str) -> str | None:
    """Resolve a named law from indexed titles, without a static alias list.

    This lets a corpus grow from 100 to 1,000+ laws without hand-maintaining a
    dictionary. Two distinctive title terms are required to avoid guessing.
    """
    query_tokens = _title_tokens(normalized_query)
    if len(query_tokens) < 2:
        return None
    try:
        from RAG.retriever.bm25_index import get_bm25_index

        docs = get_bm25_index().docs
    except Exception:
        return None
    candidates: dict[str, set[str]] = {}
    for doc in docs:
        meta = doc.metadata or {}
        law = str(meta.get("law_number") or "").strip()
        if not law or law == "unknown":
            continue
        title = str(meta.get("law_name") or meta.get("source_file") or "")
        tokens = _title_tokens(title)
        if tokens:
            candidates.setdefault(law, set()).update(tokens)
    matches = [
        (len(tokens), law) for law, tokens in candidates.items()
        if len(tokens) >= 2 and tokens.issubset(query_tokens)
    ]
    return max(matches)[1] if matches else None

class QueryMetadataExtractor:
    def __init__(self):
        # 🚀 Geliştirilmiş Regex: 2-4 haneli rakamları ve "sayılı/sayili" varyasyonlarını yakala
        # Örnek: "1076 sayılı", "4857 Sayılı", "5510 SAYILI", "213 sayili"
        self.law_pattern = re.compile(r"(\b\d{2,4}\b)\s*(?:sayılı|Sayılı|SAYILI|sayili|Sayili)", re.IGNORECASE)
        
        # Örnek: "Madde 3", "m. 12", "3. maddesi", "100 maddesine".
        # Doğal dil soruları çoğunlukla sayı-önce biçimini kullanır; ekleri
        # kaçırmak kesin sorguya komşu maddelerin karışmasına yol açar.
        self.article_pattern = re.compile(
            r"(?:Madde|madde|MADDE|m\.|m)\s*(\d+)"
            r"|(\d+)\s*\.?\s*madde(?:si|sine|sinin|nin|de|den)?",
            re.IGNORECASE,
        )
        self.compact_citation_pattern = re.compile(r"\b(\d{2,4})\s*/\s*(\d+)\b")
        self.alias_article_pattern = re.compile(r"\b(?:cmk|tck|kvkk)\s*(\d+)\b", re.IGNORECASE)
        self.law_aliases = {
            "cmk": "5271", "ceza muhakemesi kanunu": "5271",
            "tck": "5237", "turk ceza kanunu": "5237",
            "kvkk": "6698", "is kanunu": "4857",
        }
        
    def extract(self, query: str) -> Dict[str, Any]:
        """
        Sorgudan kanun ve madde numarasını çıkarır.
        ChromaDB/BM25 'where' filtrelemesi için uygun bir sözlük döndürür.
        """
        filters = {}

        # Hukuk kullanıcıları sıkça 5271/100 gibi kısa atıf girer. Normal
        # başlık/takma ad çözümlemesinden önce iki değer birlikte yakalanır.
        compact_match = self.compact_citation_pattern.search(query)
        if compact_match:
            filters["law_number"] = compact_match.group(1)
            filters["article_no"] = compact_match.group(2)
        
        # Kanun Numarasını Çıkar
        law_match = self.law_pattern.search(query)
        if law_match and "law_number" not in filters:
            filters["law_number"] = law_match.group(1)
        elif "law_number" not in filters:
            # Sadece casefold, Türkçe noktalı İ ile ASCII takma adları ayıramaz.
            # Önce fold uygulanır; farklı klavye girişleri kanun filtresini tutarlı çalıştırır.
            lowered = fold_turkish(query).casefold()
            for alias, law_number in self.law_aliases.items():
                if alias in lowered:
                    filters["law_number"] = law_number
                    break
            else:
                # Corpus tabanlı başlık eşlemesi, kırılgan manuel sözlüğü sürekli
                # büyütmeden yeni eklenen kanunları kapsar.
                inferred = _law_from_index(fold_turkish(query).casefold())
                if inferred:
                    filters["law_number"] = inferred
            
        # Madde Numarasını Çıkar
        article_match = self.article_pattern.search(query)
        alias_article_match = self.alias_article_pattern.search(query)
        if article_match and "article_no" not in filters:
            # İki farklı grup var, hangisi doluysa onu al
            art_no = article_match.group(1) or article_match.group(2)
            if art_no:
                filters["article_no"] = art_no
        elif alias_article_match and "article_no" not in filters:
            filters["article_no"] = alias_article_match.group(1)
            
        return filters

# Singleton örneği
_extractor = None

def get_query_metadata_extractor() -> QueryMetadataExtractor:
    global _extractor
    if _extractor is None:
        _extractor = QueryMetadataExtractor()
    return _extractor
