"""
Sorgu Meta Veri Çıkarıcı (Classification Agent Mantığı) - Geliştirilmiş Versiyon
-------------------------------------------------------
Kullanıcı sorgularından kanun numarası (law_number) ve madde numarasını (article_no) 
çıkararak ChromaDB ve BM25 üzerinde sıkı ön filtreleme (strict pre-filtering) yapılmasını sağlar.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Any

from RAG.retriever.text_utils import fold_turkish, tokenize


@dataclass(frozen=True)
class QueryIntent:
    """Router ve Graph-RAG için çoklu hukukî sorgu özeti."""

    law_numbers: tuple[str, ...]
    article_numbers: tuple[str, ...]
    primary_law_number: str | None
    amending_law_numbers: tuple[str, ...]
    kind: str
    needs_multiple_evidence: bool


def _title_tokens(value: str) -> set[str]:
    """Meaningful tokens in a source title; numbers and generic words vanish."""
    generic = {
        "kanun", "kanunu", "hakkinda", "pdf", "sayili", "degisiklik", "degistiren",
        "mevzuat", "liste", "listesi", "gore", "getiren", "iptal", "mulga", "yururluk",
        "yonetmeligi", "yonetmelik", "tebligi", "teblig", "tuzugu", "tuzuk",
        "ile", "yapilan", "duzenlenmesi", "iliskin", "dair", "amac", "esas", "tarih", "tarihi",
    }
    return {token for token in tokenize(fold_turkish(value).casefold(), min_len=3) if not token.isdigit() and token not in generic}


@lru_cache(maxsize=512)
def _law_from_index(normalized_query: str, *, allow_unique_token: bool = False) -> str | None:
    """Kanun adını primary-law başlıklarından çözer."""
    query_tokens = _title_tokens(normalized_query)
    if len(query_tokens) < 2:
        return None
    candidates = _law_catalog()
    if not candidates:
        return None
    matches = [
        (len(tokens), law) for law, tokens in candidates.items()
        if len(tokens) >= 2 and tokens.issubset(query_tokens)
    ]
    if matches:
        return max(matches)[1]
    if not allow_unique_token:
        return None
    owners: dict[str, set[str]] = {}
    for law, tokens in candidates.items():
        for token in tokens:
            owners.setdefault(token, set()).add(law)
    unique = {next(iter(values)) for token, values in owners.items() if token in query_tokens and len(values) == 1}
    return next(iter(unique)) if len(unique) == 1 else None


@lru_cache(maxsize=1)
def _law_catalog() -> dict[str, set[str]]:
    """Kalıcı SQLite kataloğu router'ı her soruda 18k chunk taramaktan kurtarır."""
    try:
        from RAG.metadata.legal_index import get_legal_index

        catalog = get_legal_index().law_catalog()
        if catalog:
            # SQLite kataloğu sayfa/dosya adının tüm tokenlarını tutar; router
            # için bunları eski BM25 yolu ile aynı anlamlı başlık tokenlarına
            # indirgeriz. Böylece "Türk Ceza Kanunu" için "kanunu" yazılması
            # zorunlu olmaz, fakat OHAL gibi türetilmiş kısa adlar korunur.
            cleaned_catalog: dict[str, set[str]] = {}
            for law, raw_tokens in catalog.items():
                tokens = _title_tokens(" ".join(raw_tokens))
                # legal_index kısa kısaltma üretirken "k" + sonraki kelime
                # (ör. ``kcocuk``) de ekleyebilir. Bunlar doğal başlık sözcüğü
                # olmadığı için tam başlık eşleşmesini engellememelidir.
                tokens = {
                    token for token in tokens
                    if not (len(token) > 4 and token.startswith("k") and token[1:] in tokens)
                }
                if tokens:
                    cleaned_catalog[law] = tokens
            return cleaned_catalog
    except Exception:
        pass
    # İlk pipeline tamamlanmadan da sistem çalışabilsin diye eski indekse geri dön.
    try:
        from RAG.retriever.bm25_index import get_bm25_index

        docs = get_bm25_index().docs
    except Exception:
        return {}
    candidates: dict[str, set[str]] = {}
    for doc in docs:
        meta = doc.metadata or {}
        if meta.get("source_type") != "laws":
            continue
        law = str(meta.get("law_number") or "").strip()
        if law and law != "unknown":
            candidates.setdefault(law, set()).update(_title_tokens(str(meta.get("source_file") or meta.get("law_name") or "")))
    return candidates


def reset_query_metadata_cache() -> None:
    """Corpus değişiminden sonra dinamik kanun kataloğunu da yeniler."""
    _law_from_index.cache_clear()
    _law_catalog.cache_clear()


def _coordinated_law_numbers(query: str) -> list[str]:
    """``7196 ve 7547 sayılı`` gibi ortak ekli numaraları ayrı atıflara böler."""
    numbers: list[str] = []
    pattern = re.compile(
        r"\b((?:\d{2,5}\s*(?:,|\bve\b|\bveya\b)\s*)+\d{2,5})\s*(?:sayılı|sayili)\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(query):
        for value in re.findall(r"\d{2,5}", match.group(1)):
            if value not in numbers:
                numbers.append(value)
    return numbers

class QueryMetadataExtractor:
    def __init__(self):
        # Geliştirilmiş Regex: 2-4 haneli rakamları ve "sayılı/sayili" varyasyonlarını yakala
        # Örnek: "1076 sayılı", "4857 Sayılı", "5510 SAYILI", "213 sayili"
        self.law_pattern = re.compile(r"(\b\d{2,5}\b)\s*(?:sayılı|Sayılı|SAYILI|sayili|Sayili)", re.IGNORECASE)
        # Resmî başlıklarda sık görülen "Kanun No: 5326" biçimi, sayıdan
        # sonra "sayılı" gelmediği için ayrı ele alınır.
        self.law_no_pattern = re.compile(
            r"\bkanun\s*(?:no|numarası|numarasi)\s*[:.]?\s*(\d{2,5})\b",
            re.IGNORECASE,
        )
        
        # Örnek: "Madde 3", "m. 12", "3. maddesi", "100 maddesine".
        # Doğal dil soruları çoğunlukla sayı-önce biçimini kullanır; ekleri
        # kaçırmak kesin sorguya komşu maddelerin karışmasına yol açar.
        self.article_pattern = re.compile(
            r"(?:Madde|madde|MADDE|m\.|m)\s*(\d+)"
            r"|(\d+)\s*\.?\s*madde(?:si|sine|sinin|nin|de|den)?",
            re.IGNORECASE,
        )
        self.compact_citation_pattern = re.compile(r"\b(\d{2,4})\s*/\s*(\d{1,3})(?!\s*/\s*\d{2,4})\b")
        self.alias_article_pattern = re.compile(r"\b(?:cmk|tck|kvkk)\s*(\d+)\b", re.IGNORECASE)
        self.extra_article_pattern = re.compile(r"\bek\s+madde\s*(\d+[A-Za-z]?)\b", re.IGNORECASE)
        self.law_aliases = {
            "cmk": "5271", "ceza muhakemesi kanunu": "5271",
            "tck": "5237", "turk ceza kanunu": "5237",
            "kvkk": "6698", "is kanunu": "4857",
        }

    def _compact_citation(self, query: str):
        """Mahkeme dosya numarası/tarihini kanun-madde atfından ayırır."""
        for match in self.compact_citation_pattern.finditer(query):
            prefix = query[max(0, match.start() - 12):match.start()]
            if re.search(r"\b[EKek]\s*\.?\s*:\s*$", prefix):
                continue
            if match.start() and query[match.start() - 1] == "/":
                continue
            return match
        return None
        
    def extract(self, query: str) -> Dict[str, Any]:
        """
        Sorgudan kanun ve madde numarasını çıkarır.
        ChromaDB/BM25 'where' filtrelemesi için uygun bir sözlük döndürür.
        """
        filters = {}

        # Hukuk kullanıcıları sıkça 5271/100 gibi kısa atıf girer. Normal
        # başlık/takma ad çözümlemesinden önce iki değer birlikte yakalanır.
        compact_match = self._compact_citation(query)
        if compact_match:
            filters["law_number"] = compact_match.group(1)
            filters["article_no"] = compact_match.group(2)
        
        # Kısa atıf ve bilinen takma adlar kesin filtrelerdir.
        lowered = fold_turkish(query).casefold()
        if "law_number" not in filters:
            alias_law = next((law for alias, law in self.law_aliases.items() if alias in lowered), None)
            named_law = _law_from_index(lowered)
            unique_named_law = _law_from_index(lowered, allow_unique_token=True)
            raw_law = self.law_pattern.search(query)
            law_no = self.law_no_pattern.search(query)
            # Kullanıcının yazdığı "Kanun No" en güvenli işarettir; katalog
            # tahmini ve kısa adlar bunu hiçbir zaman geçemez.
            if law_no:
                filters["law_number"] = law_no.group(1)
            elif raw_law:
                # Kullanıcının açıkça yazdığı numara, başlık/anahtar kelime
                # kataloğundan çıkarılan hiçbir tahminden zayıf değildir.
                filters["law_number"] = raw_law.group(1)
            elif named_law:
                filters["law_number"] = named_law
            elif alias_law:
                filters["law_number"] = alias_law
            elif unique_named_law:
                filters["law_number"] = unique_named_law
            
        # Madde Numarasını Çıkar
        extra_article_match = self.extra_article_pattern.search(query)
        article_match = self.article_pattern.search(query)
        alias_article_match = self.alias_article_pattern.search(query)
        if extra_article_match and "article_no" not in filters:
            filters["article_no"] = f"Ek Madde {extra_article_match.group(1)}"
        elif article_match and "article_no" not in filters:
            # İki farklı grup var, hangisi doluysa onu al
            art_no = article_match.group(1) or article_match.group(2)
            if art_no:
                filters["article_no"] = art_no
        elif alias_article_match and "article_no" not in filters:
            filters["article_no"] = alias_article_match.group(1)
            
        return filters

    def extract_strict_filters(self, query: str) -> Dict[str, Any]:
        """Yalnız açık hukukî atıflardan güvenli ön-filtre üretir.

        Başlıktaki tek bir kelimeden çıkarılan kanun numarası faydalı bir
        *aday sinyalidir*, fakat aramayı tek kanuna kapatacak kadar kesin
        değildir. Bu metod sadece kullanıcı tarafından yazılmış numara,
        kısa atıf, bilinen kısaltma veya en az iki başlık kelimesinin tam
        eşleşmesi halinde filtre döndürür.
        """
        filters: Dict[str, Any] = {}
        compact_match = self._compact_citation(query)
        if compact_match:
            return {
                "law_number": compact_match.group(1),
                "article_no": compact_match.group(2),
            }

        lowered = fold_turkish(query).casefold()
        alias_law = next((law for alias, law in self.law_aliases.items() if alias in lowered), None)
        raw_laws = list(self.law_pattern.finditer(query))
        law_no_match = self.law_no_pattern.search(query)
        named_law = _law_from_index(lowered)
        unique_named_law = _law_from_index(lowered, allow_unique_token=True)

        # Değişiklik sorusunda tek "X sayılı Kanun" çoğu zaman hedef kanun
        # değil, değiştiren düzenlemedir. Hedef açık değilse geniş arama yapılır.
        amendment_words = ("degisiklik", "degistir", "etkile", "iptal", "mulga", "khk", "yururluk", "degisiklik cetveli")
        is_amendment = any(word in lowered for word in amendment_words)
        explicit_target = bool(
            raw_laws
            and (
                not is_amendment
                or (
                    raw_laws[0].start() <= 3
                    and (
                        re.search(
                            r"kanun(?:['’]?[a-z]+)?\s+(?:kapsaminda|uyarinca|maddesi|madde|ek|mevzuat)",
                            lowered[raw_laws[0].end():],
                        )
                        or (
                            len(raw_laws) >= 2
                            and "kanun" in lowered[raw_laws[0].end():raw_laws[1].start()]
                        )
                    )
                )
            )
        )
        if law_no_match:
            filters["law_number"] = law_no_match.group(1)
        elif raw_laws:
            # Açık numaralı atıf, katalogdan türetilmiş adaydan önceliklidir.
            filters["law_number"] = raw_laws[0].group(1)
        elif explicit_target:
            # Açık numaralı hedef atıf, başlıktan türetilmiş bir adaydan güçlüdür.
            filters["law_number"] = raw_laws[0].group(1)
        elif alias_law:
            filters["law_number"] = alias_law
        elif named_law:
            filters["law_number"] = named_law
        elif "kanun" in lowered and unique_named_law:
            filters["law_number"] = unique_named_law

        # Madde numarası tek başına yüzlerce farklı kanunda bulunabilir; bu
        # yüzden yalnız kesin kanun filtresiyle birlikte kullanılabilir.
        if "law_number" in filters:
            extra_article_match = self.extra_article_pattern.search(query)
            article_match = self.article_pattern.search(query)
            alias_article_match = self.alias_article_pattern.search(query)
            if extra_article_match:
                filters["article_no"] = f"Ek Madde {extra_article_match.group(1)}"
            elif article_match:
                filters["article_no"] = article_match.group(1) or article_match.group(2)
            elif alias_article_match:
                filters["article_no"] = alias_article_match.group(1)
        return {key: value for key, value in filters.items() if value}

    def extract_intent(self, query: str) -> QueryIntent:
        """Tek filtre yerine tüm atıfları ve kanıt gereksinimini çıkarır."""
        normalized = fold_turkish(query).casefold()
        legacy = self.extract(query)
        raw_laws = [match.group(1) for match in self.law_pattern.finditer(query)]
        raw_laws.extend(match.group(1) for match in self.law_no_pattern.finditer(query))
        raw_laws.extend(_coordinated_law_numbers(query))
        compact = self._compact_citation(query)
        if compact:
            raw_laws.insert(0, compact.group(1))
        primary = str(legacy.get("law_number") or "").strip() or None
        laws = []
        for value in ([primary] if primary else []) + raw_laws:
            if value and value not in laws:
                laws.append(value)
        articles = []
        for match in self.extra_article_pattern.finditer(query):
            value = f"Ek Madde {match.group(1)}"
            if value not in articles:
                articles.append(value)
        for match in self.article_pattern.finditer(query):
            value = match.group(1) or match.group(2)
            if value and value not in articles:
                articles.append(value)
        if legacy.get("article_no") and str(legacy["article_no"]) not in articles:
            articles.insert(0, str(legacy["article_no"]))

        amendment_terms = (
            "degisiklik", "degistir", "etkile", "mevzuat listesi", "mevzuat tablosu",
            "degisiklik tablosu", "degisiklik cetveli", "iptal", "mulga", "yururluge giris",
        )
        comparison_terms = ("arasindaki fark", "farklar", "fark nedir", "karsilastir", "hangisi", "ayrimi")
        multi_terms = ("ayrica", "diger yandan", "bunun yaninda", "hem ", " ile birlikte")
        is_amendment = any(term in normalized for term in amendment_terms)
        is_comparison = any(term in normalized for term in comparison_terms)
        is_temporal = any(term in normalized for term in ("ne zaman", "yururluk", "kabul tarihi", "tarih"))
        is_authority = any(term in normalized for term in ("hangi kurum", "hangi bakan", "hangi vekalet", "yetkili merci", "yetkili"))
        is_sanction = any(term in normalized for term in ("ceza", "usulsuzluk", "para cezasi", "hapis", "tutar"))
        is_condition = any(term in normalized for term in ("sart", "kosul", "hangi hallerde", "zorunlu"))
        # "uyarınca" ve "dayanak" tek başına bir kanunlar-arası ilişki
        # değildir; sıradan madde sorularını gereksiz Graph-RAG yoluna sokar.
        is_relation = any(term in normalized for term in ("atif", "baglanti", "iliski"))
        if is_amendment:
            kind = "amendment"
        elif is_comparison:
            kind = "comparison"
        elif len(laws) > 1 or is_relation:
            kind = "multi_law_relation"
        elif is_temporal:
            kind = "temporal"
        elif is_authority:
            kind = "authority"
        elif is_sanction:
            kind = "sanction"
        elif is_condition:
            kind = "condition"
        else:
            kind = "general"
        requires_multiple = is_comparison or (len(laws) > 1 and not is_amendment) or any(term in normalized for term in multi_terms)
        amending = tuple(value for value in laws if is_amendment and value != primary)
        return QueryIntent(tuple(laws), tuple(articles), primary, amending, kind, requires_multiple)

# Singleton örneği
_extractor = None

def get_query_metadata_extractor() -> QueryMetadataExtractor:
    global _extractor
    if _extractor is None:
        _extractor = QueryMetadataExtractor()
    return _extractor
