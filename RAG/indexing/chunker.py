"""
KutupAI - Gelişmiş Hukuki Metin Bölümleyici (Indexing Pipeline Uyumlu)
----------------------------------------------------------------------
Bu modül, RAG/indexing/indexer.py tarafından kullanılan RawChunk yapısını korurken,
arka planda KutupAI'ın gelişmiş Madde/Fıkra/Bent algılama ve temizleme mantığını kullanır.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any

from RAG.configuration.rag_config_loader import chunking_config

@dataclass
class RawChunk:
    """Eski indexer.py ile uyumluluk için ham chunk yapısı."""
    text: str
    article_number: str  
    part_index: int      

# KutupAI düzenli ifadeleri
ARTICLE_PATTERN = re.compile(r"(?im)(?<!\w)Madde\s+(\d+)\s*(?:[-–—.:])?")
ADDITIONAL_ARTICLE_PATTERN = re.compile(r"(?im)(?<!\w)(Ek\s+Madde|Geçici\s+Madde|Muvakkat\s+Madde)\s*(\d+)?\s*(?:[-–—.:])?")
PARAGRAPH_PATTERN = re.compile(r"(?m)^\s*\((\d+)\)\s+")
CLAUSE_PATTERN = re.compile(r"(?m)(?:^|\s)([a-zA-ZçğıöşüÇĞİÖŞÜ])\)\s+")

def normalize_whitespace(text: str) -> str:
    if not text: return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r"\n?\s*[—\-]{5,}\s*\n?", "\n", text)
    text = re.sub(r"(?m)^\s*\d{1,4}(?:-\d{1,2})?\s*$", "", text)
    text = re.sub(r"\s*\((\d+)\)\s*\n", r" (\1) ", text)
    return normalize_whitespace(text)

def _split_by_article(full_text: str) -> List[Dict[str, Any]]:
    """Metni akıllıca maddelere ayırır (KutupAI Mantığı)."""
    text = clean_text(full_text)
    if not text: return [{"content": text, "article_number": "", "type": "legal_document"}]

    # Tabloları atla.
    table_match = re.search(r"(?i)(DEĞİŞİKLİKLER CETVELİ|YÜRÜRLÜKTEN KALDIRDIĞI KANUN|YÜRÜRLÜĞE GİRİŞ TARİHİNİ GÖSTER)", text)
    if table_match:
        text = text[:table_match.start()].strip()

    normal_matches = list(ARTICLE_PATTERN.finditer(text))
    additional_matches = list(ADDITIONAL_ARTICLE_PATTERN.finditer(text))
    all_matches = sorted(normal_matches + additional_matches, key=lambda m: m.start())

    if not all_matches:
        return [{"content": text, "article_number": "", "type": "legal_document"}]

    articles = []
    first_start = all_matches[0].start()
    if first_start > 0:
        articles.append({"content": text[:first_start].strip(), "article_number": "", "type": "preamble"})

    for i, match in enumerate(all_matches):
        start = match.start()
        end = all_matches[i + 1].start() if i + 1 < len(all_matches) else len(text)
        art_text = text[start:end].strip()
        
        art_no = ""
        if ARTICLE_PATTERN.match(text, start):
            art_no = ARTICLE_PATTERN.match(text, start).group(1)
        elif ADDITIONAL_ARTICLE_PATTERN.match(text, start):
            m = ADDITIONAL_ARTICLE_PATTERN.match(text, start)
            art_no = f"{m.group(1).strip()} {m.group(2)}".strip() if m.group(2) else m.group(1).strip()
            
        articles.append({"content": art_text, "article_number": art_no, "type": "madde"})

    return articles

def _split_long_chunk(content: str, article_number: str, max_size: int, overlap: int) -> List[RawChunk]:
    """Uzun maddeleri fıkra/bent yapısına göre veya cümle bazlı böler."""
    if len(content) <= max_size:
        return [RawChunk(text=content, article_number=article_number, part_index=0)]

    # Yapısal bölme dene (Bent veya Fıkra)
    sections = []
    clause_matches = list(CLAUSE_PATTERN.finditer(content))
    if len(clause_matches) > 1:
        if clause_matches[0].start() > 0: sections.append(content[:clause_matches[0].start()].strip())
        for i, m in enumerate(clause_matches):
            start = m.start()
            end = clause_matches[i+1].start() if i+1 < len(clause_matches) else len(content)
            sections.append(content[start:end].strip())
    else:
        para_matches = list(PARAGRAPH_PATTERN.finditer(content))
        if len(para_matches) > 1:
            if para_matches[0].start() > 0: sections.append(content[:para_matches[0].start()].strip())
            for i, m in enumerate(para_matches):
                start = m.start()
                end = para_matches[i+1].start() if i+1 < len(para_matches) else len(content)
                sections.append(content[start:end].strip())

    if not sections:
        sections = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]

    raw_chunks = []
    current_chunk = ""
    part_index = 0

    for sec in sections:
        if len(sec) > max_size:
            if current_chunk:
                raw_chunks.append(RawChunk(text=current_chunk.strip(), article_number=article_number, part_index=part_index))
                part_index += 1
                current_chunk = ""
            sentences = re.split(r'(?<=[.!?])\s+', sec)
            temp = ""
            for s in sentences:
                if len(temp) + len(s) + 1 <= max_size:
                    temp += (" " + s).strip()
                else:
                    if temp:
                        raw_chunks.append(RawChunk(text=temp.strip(), article_number=article_number, part_index=part_index))
                        part_index += 1
                    temp = s
            if temp: current_chunk = temp
            continue

        candidate = f"{current_chunk}\n\n{sec}" if current_chunk else sec
        if len(candidate) <= max_size:
            current_chunk = candidate
        else:
            if current_chunk:
                raw_chunks.append(RawChunk(text=current_chunk.strip(), article_number=article_number, part_index=part_index))
                part_index += 1
            if overlap > 0 and len(current_chunk) > overlap:
                ov = current_chunk[-overlap:]
                sp = ov.find(" ")
                if sp != -1: ov = ov[sp+1:]
                current_chunk = ov + "\n\n" + sec
            else:
                current_chunk = sec

    if current_chunk:
        raw_chunks.append(RawChunk(text=current_chunk.strip(), article_number=article_number, part_index=part_index))

    return raw_chunks

def chunk_document(full_text: str) -> List[RawChunk]:
    """Ana giriş noktası: Eski indexer.py ile tam uyumlu çalışır."""
    max_size = getattr(chunking_config, 'max_chunk_size_chars', chunking_config.chunk_size)
    overlap = getattr(chunking_config, 'chunk_overlap_chars', chunking_config.chunk_overlap)
    
    articles = _split_by_article(full_text)
    final_chunks: List[RawChunk] = []
    
    for art in articles:
        chunks = _split_long_chunk(art["content"], art["article_number"], max_size, overlap)
        final_chunks.extend(chunks)
        
    return final_chunks
