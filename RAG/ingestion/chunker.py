"""
KutupAI - Gelişmiş Hukuki Metin Bölümleyici (Advanced Legal-Aware Chunker)
--------------------------------------------------------------------------
Varsayılan RecursiveCharacterTextSplitter yerine, Türk hukuk metinlerinin 
yapısını (Madde, Fıkra, Bent) anlayan ve akıllı bölme yapan modül.

Özellikler:
1. Madde, Ek Madde, Geçici Madde tespiti.
2. Fıkra (1, 2, 3) ve Bent (a, b, c) bazlı yapısal bölme.
3. Mülga (yürürlükten kaldırılmış) maddelerin korunması.
4. Kanun sonundaki değişiklik cetvelleri ve listelerin (Validation) filtrelenmesi.
5. Dipnotların (1, 2) akıllıca metne yedirilmesi.
6. 🚀 Çok kısa veya anlamsız chunk'ların (Örn: "Ek", "Geçici", tablo başlıkları) filtrelenmesi.
"""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document

# Projenin kendi konfigürasyonunu kullanarak chunk boyutlarını dinamik tutuyoruz
from RAG.configuration.rag_config_loader import chunking_config

# ============================================================
# Regex Tanımları (Türk Hukuk Yapısı İçin)
# ============================================================

# Standart madde başlangıçları (Örn: "Madde 3-", "MADDE 5 –", "Madde 7 7 –")
ARTICLE_PATTERN = re.compile(
    r"(?im)(?<!\w)Madde\s+"
    r"(\d+(?:\s+\d+)?)" # Birden fazla rakamı da yakala (Örn: Madde 7 7)
    r"\s*(?:[-–—.:])?"
)

# Özel madde türleri (Örn: "Ek Madde 1", "Geçici Madde 5", "Madde 10 ilâ 15")
ADDITIONAL_ARTICLE_PATTERN = re.compile(
    r"(?im)(?<!\w)"
    r"(Ek\s+Madde|Geçici\s+Madde|Muvakkat\s+Madde|Madde\s+\d+\s+ilâ\s+\d+)"
    r"\s*(\d+)?"
    r"\s*(?:[-–—.:])?"
)

# Fıkra tespiti (Örn: "(1) ", "(2) ")
PARAGRAPH_PATTERN = re.compile(r"(?m)^\s*\((\d+)\)\s+")

# Bent tespiti (Örn: "a) ", "b) ", "c) ", "1 - ", "2 - ")
CLAUSE_PATTERN = re.compile(r"(?m)(?:^|\s)([a-zA-ZçğıöşüÇĞİÖŞÜ])\)\s+|(?:^|\s)(\d+)\s*[-–]\s+")
PAGE_MARKER_PATTERN = re.compile(r"\[\[RAG_PAGE:(\d+)\]\]")

# ============================================================
# Metin Temizleme Fonksiyonları (Data Validation & Cleaning)
# ============================================================

def normalize_whitespace(text: str) -> str:
    """Gereksiz boşlukları ve satır sonlarını standartlaştırır."""
    if not text: return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_text(text: str) -> str:
    """
    OCR veya PDF okuma sonrası oluşan gürültüyü temizler.
    """
    if not text: return ""
    
    # 1. Sayfa ayırıcı çizgileri temizle (Örn: ——————————)
    text = re.sub(r"\n?\s*[—\-]{5,}\s*\n?", "\n", text)
    
    # 2. Tek başına kalan sayfa numaralarını temizle (Örn: 749, 750, 760-1)
    text = re.sub(r"(?m)^\s*\d{1,4}(?:-\d{1,2})?\s*$", "", text)
    
    # 3. Satır sonlarındaki dipnotları (1), (2) metnin içine akıcı şekilde yedir
    text = re.sub(r"\s*\((\d+)\)\s*\n", r" (\1) ", text)
    
    return normalize_whitespace(text)


def _page_at_offset(text: str, offset: int, fallback: Any = None) -> int | None:
    """Find the original PDF page active at a normalized-text offset."""
    page = int(fallback) if isinstance(fallback, (int, float)) else None
    for match in PAGE_MARKER_PATTERN.finditer(text):
        if match.start() > offset:
            break
        page = int(match.group(1))
    return page

# ============================================================
# Madde Tespiti ve Yapısal Bölme (Structural Extraction)
# ============================================================

def split_articles(document: Document) -> List[Dict[str, Any]]:
    """
    Belgenin tamamını tarayarak her bir maddeyi (ve giriş/preamble kısmını) ayırır.
    """
    text = clean_text(document.page_content)
    if not text: return []

    # 🚀 VALIDATION: Kanun sonundaki devasa değişiklik cetvellerini ve listeleri atla
    # Bu kısımlar RAG için gürültü oluşturur (Örn: 1076, 7068 sayılı kanunların sonundaki cetveller)
    table_match = re.search(
        r"(?i)("
        r"DEĞİŞİKLİKLER CETVELİ|"
        r"YÜRÜRLÜKTEN KALDIRDIĞI KANUN|"
        r"YÜRÜRLÜĞE GİRİŞ TARİHİNİ GÖSTER|"
        r"ÇEŞİTLİ MEVZUAT İLE YAPILAN|"
        r"EK VE DEĞİŞİKLİK YAPAN MEVZUATIN|"
        r"EK VE DEĞİŞİKLİK GETİREN|"
        r"GÖSTERİR LİSTE|"
        r"Kanun Yürürlüğe\s+No\.|"
        r"Farklı tarihte Yürürlüğe giren"
        r")", 
        text
    )
    if table_match:
        text = text[:table_match.start()].strip()

    normal_matches = list(ARTICLE_PATTERN.finditer(text))
    additional_matches = list(ADDITIONAL_ARTICLE_PATTERN.finditer(text))
    
    all_matches = normal_matches + additional_matches
    all_matches.sort(key=lambda match: match.start())

    # Aynı pozisyonda başlayan eşleşmeleri tekilleştir
    unique_matches = []
    seen_positions = set()
    for match in all_matches:
        if match.start() not in seen_positions:
            seen_positions.add(match.start())
            unique_matches.append(match)
    all_matches = unique_matches

    # Hiç madde bulunamadıysa tüm metni tek parça olarak döndür
    if not all_matches:
        return [{"article_no": None, "article_type": "legal_document", "content": text, "start_offset": 0}]

    articles = []
    
    # İlk maddeden önceki kısım (Kanun adı, amaç, kapsam vb. - Preamble)
    first_article_start = all_matches[0].start()
    if first_article_start > 0:
        preamble = text[:first_article_start].strip()
        if preamble:
            articles.append({"article_no": None, "article_type": "preamble", "content": preamble, "start_offset": 0})

    # Her bir maddeyi sınırlarına göre kes ve meta verisini çıkar
    for index, match in enumerate(all_matches):
        start = match.start()
        end = all_matches[index + 1].start() if index + 1 < len(all_matches) else len(text)
        
        article_text = text[start:end].strip()
        if not article_text: continue

        article_no = None
        article_type = "madde"

        normal_match = ARTICLE_PATTERN.match(text, start)
        if normal_match:
            article_no = normal_match.group(1).replace(" ", "") # Boşlukları temizle (Örn: "7 7" -> "77")
            article_type = "madde"
        else:
            additional_match = ADDITIONAL_ARTICLE_PATTERN.match(text, start)
            if additional_match:
                prefix = additional_match.group(1).strip()
                number = additional_match.group(2)
                
                # "Madde 10 ilâ 15" gibi aralık ifadelerini yakala
                if "ilâ" in prefix or "ila" in prefix:
                    article_no = prefix.replace("Madde ", "").replace("madde ", "")
                else:
                    article_no = f"{prefix} {number}" if number else prefix
                    
                prefix_lower = prefix.lower()
                if "geçici" in prefix_lower: article_type = "gecici_madde"
                elif "ek" in prefix_lower: article_type = "ek_madde"
                elif "muvakkat" in prefix_lower: article_type = "muvakkat_madde"

        articles.append({
            "article_no": article_no,
            "article_type": article_type,
            "content": article_text,
            "start_offset": start,
        })

    return articles

def _extract_structured_sections(content: str) -> List[str]:
    """
    Uzun bir maddenin içindeki fıkra (1, 2) veya bentleri (a, b) tespit edip alt parçalara ayırır.
    """
    # Önce bentlere (a, b, c) veya numaralı listelere (1 -, 2 -) göre bölmeyi dene
    clause_matches = list(CLAUSE_PATTERN.finditer(content))
    if len(clause_matches) > 1:
        sections = []
        if clause_matches[0].start() > 0:
            prefix = content[:clause_matches[0].start()].strip()
            if prefix: sections.append(prefix)
            
        for i, match in enumerate(clause_matches):
            start = match.start()
            end = clause_matches[i + 1].start() if i + 1 < len(clause_matches) else len(content)
            section = content[start:end].strip()
            if section: sections.append(section)
        return sections

    # Bent yoksa fıkralara (1, 2, 3) göre bölmeyi dene
    paragraph_matches = list(PARAGRAPH_PATTERN.finditer(content))
    if len(paragraph_matches) > 1:
        sections = []
        if paragraph_matches[0].start() > 0:
            prefix = content[:paragraph_matches[0].start()].strip()
            if prefix: sections.append(prefix)
            
        for i, match in enumerate(paragraph_matches):
            start = match.start()
            end = paragraph_matches[i + 1].start() if i + 1 < len(paragraph_matches) else len(content)
            section = content[start:end].strip()
            if section: sections.append(section)
        return sections

    # Hiçbiri yoksa çift satır sonuna göre böl
    return [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]


def _hard_split(text: str, max_size: int) -> List[str]:
    """Split a pathological long sentence without losing any text.

    OCR'd legal tables and gazette notices can contain thousands of characters
    without a sentence delimiter.  Passing one of these to the embedding model
    pads the entire batch to that exceptional length, severely slowing GPU
    indexing.  Prefer a whitespace boundary, but always enforce ``max_size``.
    """
    remaining = text.strip()
    parts: List[str] = []
    while len(remaining) > max_size:
        cut = remaining.rfind(" ", 0, max_size + 1)
        if cut < max_size // 2:
            cut = max_size
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts

def chunk_article(article: Dict[str, Any], max_size: int, overlap_size: int) -> List[Dict[str, Any]]:
    """
    Tek bir maddeyi, maksimum boyutu aşmayacak şekilde akıllıca parçalara (chunk) ayırır.
    """
    article_no = article.get("article_no")
    article_type = article.get("article_type")
    content = article.get("content", "")

    if not content: return []

    # Madde zaten kısa ise direkt döndür
    if len(content) <= max_size:
        return [{"article_no": article_no, "article_type": article_type, "content": content}]

    sections = _extract_structured_sections(content)
    raw_chunks = []
    current_chunk = ""

    for section in sections:
        # Bir bölüm bile max_size'dan büyükse cümle cümle böl
        if len(section) > max_size:
            if current_chunk:
                raw_chunks.append(current_chunk.strip())
                current_chunk = ""
            
            sentences = re.split(r'(?<=[.!?])\s+', section)
            temp_chunk = ""
            for sent in sentences:
                # Tek OCR/tablo "cümlesi" ayarlı chunk boyutunu aşabilir.
                # Bu tür metinlerin sınırı atlamasına izin verilmez.
                if len(sent) > max_size:
                    if temp_chunk:
                        raw_chunks.append(temp_chunk.strip())
                        temp_chunk = ""
                    raw_chunks.extend(_hard_split(sent, max_size))
                    continue
                if len(temp_chunk) + len(sent) + 1 <= max_size:
                    temp_chunk += (" " + sent).strip()
                else:
                    if temp_chunk: raw_chunks.append(temp_chunk.strip())
                    temp_chunk = sent
            if temp_chunk: raw_chunks.append(temp_chunk.strip())
            continue

        candidate = f"{current_chunk}\n\n{section}" if current_chunk else section
        
        if len(candidate) <= max_size:
            current_chunk = candidate
        else:
            if current_chunk:
                raw_chunks.append(current_chunk.strip())
            
            # Overlap (örtüşme) mantığı: Bağlamı kaybetmemek için önceki chunk'ın sonundan biraz al
            if overlap_size > 0 and len(current_chunk) > overlap_size:
                overlap_text = current_chunk[-overlap_size:]
                space_idx = overlap_text.find(" ")
                if space_idx != -1:
                    overlap_text = overlap_text[space_idx+1:]
                current_chunk = overlap_text + "\n\n" + section
            else:
                current_chunk = section

    if current_chunk:
        raw_chunks.append(current_chunk.strip())

    # Çok küçük chunk'ları birleştirme (Min Chunk Size Logic)
    final_chunks = []
    min_chunk_size = 250 # 250 karakterden küçük parçaları yanındakiyle birleştir
    for raw_chunk in raw_chunks:
        for chunk in _hard_split(raw_chunk, max_size):
            if not final_chunks:
                final_chunks.append(chunk)
                continue

            if len(chunk) < min_chunk_size and len(final_chunks[-1]) + len(chunk) + 2 <= max_size:
                final_chunks[-1] = final_chunks[-1] + "\n\n" + chunk
            else:
                final_chunks.append(chunk)

    return [
        {"article_no": article_no, "article_type": article_type, "content": chunk}
        for chunk in final_chunks if chunk.strip()
    ]

# ============================================================
# Ana Fonksiyonlar (LangChain Uyumlu)
# ============================================================

_LAW_NUMBER_HEADER_RE = re.compile(
    r"\bKanun\s*(?:Numarası|Numarasi|No\.?|N[oº])\s*[:\-]?\s*(\d{2,8})\b",
    re.IGNORECASE,
)
_NUMBERED_LAW_TITLE_RE = re.compile(r"\b(\d{2,8})\s+sayılı\s+kanun\b", re.IGNORECASE)


def _resolve_law_number(metadata: Dict[str, Any], text: str = "") -> str:
    """Resolve a law number without guessing.

    A sidecar file can add metadata, but an erroneous global ``law_number``
    must never relabel every law in the corpus.  Legal corpus filenames begin
    with their official number (for example ``1076_Yedek Subaylar...pdf``), so
    prefer that stable identifier. For files without one, use explicit sidecar
    metadata, then inspect the document header for an official law number.
    """
    source_file = str(metadata.get("source_file") or metadata.get("source") or "")
    filename_match = re.match(r"\s*(\d{2,8})(?=[_\s.-]|$)", Path(source_file).name)
    if filename_match:
        return filename_match.group(1)

    value = metadata.get("law_number", metadata.get("law_no", "unknown"))
    value = str(value).strip()
    if value and value.lower() != "unknown":
        return value

    # Belgenin başında genellikle "Kanun Numarası: 4857" yer alır. Metin içindeki
    # başka kanun atıflarını kaynak kanun sanmamak için yalnız başlık incelenir.
    header = (text or "")[:4000]
    header_match = _LAW_NUMBER_HEADER_RE.search(header)
    if header_match:
        return header_match.group(1)
    title_match = _NUMBERED_LAW_TITLE_RE.search(header)
    if title_match:
        return title_match.group(1)
    return "unknown"


def _build_chunk_id(metadata: Dict[str, Any], article_no: Any, chunk_index: int, content: str) -> str:
    """Create a stable ID that cannot collide across PDF pages or re-ingests."""
    law_no = _resolve_law_number(metadata)
        
    article_id = "preamble" if article_no is None else re.sub(r"\s+", "_", str(article_no))
    source = str(metadata.get("source_file") or metadata.get("source") or "unknown")
    normalized = " ".join(content.split())
    digest = hashlib.sha256(f"{source}|{article_id}|{normalized}".encode("utf-8")).hexdigest()[:12]
    return f"{law_no}_{article_id}_{chunk_index:05d}_{digest}"

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Ingestion pipeline tarafından çağrılan ana giriş noktası.
    Varsayılan metin bölücü yerine bu akıllı hukuk bölücüyü kullanır.
    """
    if not documents:
        return []

    all_chunks = []
    
    max_size = chunking_config.chunk_size
    overlap_size = chunking_config.chunk_overlap

    print(f"Legal-aware chunking: max_size={max_size}, overlap={overlap_size}")

    for document in documents:
        document_metadata = deepcopy(document.metadata)
        normalized_document_text = clean_text(document.page_content)
        document_metadata["law_number"] = _resolve_law_number(
            document_metadata,
            document.page_content,
        )
        articles = split_articles(document)

        for article in articles:
            chunks = chunk_article(article, max_size, overlap_size)
            search_from = int(article.get("start_offset", 0))

            for chunk_index, chunk in enumerate(chunks):
                metadata = deepcopy(document_metadata)
                article_no = chunk.get("article_no")
                article_type = chunk.get("article_type")
                content_with_markers = chunk.get("content", "")
                # Özel işaretler silinmeden chunk normalize kaynakta bulunur.
                # Böylece madde birden çok PDF sayfasını geçse de sayfa bilgisi doğru kalır.
                position = normalized_document_text.find(content_with_markers, search_from)
                if position < 0:
                    position = normalized_document_text.find(content_with_markers)
                if position >= 0:
                    search_from = position + 1  # Örtüşme önceki chunk bitiminden önce başlayabilir.
                fallback_page = document_metadata.get("page_start") or document_metadata.get("page")
                page_start = _page_at_offset(normalized_document_text, max(position, 0), fallback_page)
                # Sonraki madde sayfa işaretinden hemen sonra başlayabilir; işaret
                # mevcut chunk'a değil, sonraki maddeye aittir.
                content_for_end = re.sub(
                    r"(?:\s*\[\[RAG_PAGE:\d+\]\])+\s*$", "", content_with_markers
                ).rstrip()
                page_end = _page_at_offset(
                    normalized_document_text,
                    max(position, 0) + max(len(content_for_end) - 1, 0),
                    fallback_page,
                )
                content = PAGE_MARKER_PATTERN.sub("", content_with_markers).strip()

                chunk_id = _build_chunk_id(metadata, article_no, chunk_index, content)

                # 🚀 KRİTİK: Hybrid Retriever'daki Metadata Filtrelemesi için law_number ve article_no string olarak ekleniyor
                law_no = _resolve_law_number(metadata)

                metadata.update({
                    "law_number": str(law_no), # Hibrit arama filtresi için
                    "article_no": str(article_no) if article_no else None, # Hibrit arama filtresi için
                    "article_type": article_type,
                    "chunk_index": chunk_index,
                    "chunk_length": len(content),
                    "content_type": "legal_document",
                    "chunk_id": chunk_id,
                    "page": page_start,
                    "page_start": page_start,
                    "page_end": page_end,
                })

                # 🚀 FİLTRELEME: Çok kısa veya anlamsız chunk'ları atla (Örn: sadece "Ek", "Geçici", tablo başlıkları)
                # Mülga maddeler genellikle 40-50 karakterdir, bu yüzden 20 sınırı güvenlidir.
                if len(content.strip()) > 30:
                    all_chunks.append(Document(page_content=content, metadata=metadata))

    print(f"Legal-aware chunking complete: {len(documents)} documents -> {len(all_chunks)} chunks.")
    return all_chunks

def get_text_splitter():
    raise NotImplementedError("Doğrudan metin bölücü kullanımdan kaldırıldı. Hukuki farkındalık için split_documents() kullanın.")
