"""Değişiklik, iptal ve yürürlük cetvelleri için yapılandırılmış kanıt deposu."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from RAG.configuration.rag_config_loader import amendment_ledger_config, documents_config
from RAG.retriever.text_utils import fold_turkish
from RAG.vector_store.vector_store_interface import SearchResult

_DATE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
_LAW_FILE = re.compile(r"^(\d{3,4})[_ -]")
_NUMBER = re.compile(r"\b\d{3,4}\b")
_STRONG_HINTS = (
    "degisiklik", "degistir", "etkile", "iptal", "mulga", "mevzuat listesi",
    "mevzuat tablosu", "degisiklik tablosu", "cetvel",
    "khk", "anayasa mahkemesi", "yuksek mahkeme", "gecersiz kil", "iptal karari",
)
_CONTEXT_HINTS = ("kanun yoluyla", "sayili kanun ile", "ile yapilan duzenleme", "yururluge giris")
_TEXT_TABLE_ROW = re.compile(
    r"^\s*(?:\d+\.\s*)?(?P<source>(?:KHK\s*/\s*)?\d{3,4})\s+"
    r"(?:\d{1,2}/\d{1,2}/\d{4}\s+)?"
    r"(?P<effective>\d{1,2}/\d{1,2}/\d{4}|-)\s+(?:-\s*)?(?P<articles>.+?)\s*$",
    re.IGNORECASE,
)


def _compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _law_number(path: Path) -> str:
    match = _LAW_FILE.match(path.name)
    return match.group(1) if match else "unknown"


def _articles(value: str) -> list[str]:
    values = re.findall(r"(?:Ek\s+|Geçici\s+)?Madde\s+\d+[A-Za-z]?|\b\d+[A-Za-z]?\b", value, re.IGNORECASE)
    return list(dict.fromkeys(values))


def _date(value: str) -> str | None:
    dates = _DATE.findall(value)
    if not dates:
        return None
    raw = dates[-1] if len(dates) > 1 and "yürürlüğ" in value.casefold() else dates[0]
    day, month, year = raw.split("/")
    return f"{year}-{int(month):02d}-{int(day):02d}"


def _legal_effect(*values: str) -> str:
    """Cetvel satırını tahmin etmeden Türkçe hukukî etki etiketiyle sınıflandırır."""
    text = fold_turkish(" ".join(values)).casefold()
    if "anayasa mahkem" in text and ("iptal" in text or "gecersiz" in text):
        return "anayasa_mahkemesi_iptali"
    if "mulga" in text or "yururlukten kaldir" in text:
        return "yururlukten_kaldirma"
    if "ek madde" in text or "ilave" in text:
        return "ekleme"
    if "degisik" in text or "degistir" in text:
        return "degisiklik"
    return "degisiklik_veya_iptal_cetvel_kaydi"


def _is_table(table: list[list[str | None]]) -> bool:
    header = " ".join(_compact(cell) for row in table[:2] for cell in row).casefold()
    return sum(term in header for term in ("değiştiren", "değiştirilen", "iptal eden", "yürürlüğe giriş")) >= 2


def _text_table_rows(page_text: str) -> list[tuple[str, str, str]]:
    """Çizgisi olmayan PDF tablolarını metin satırlarından güvenle okur.

    Bazı Resmî Gazete PDF'lerinde tablo hücreleri vardır fakat çizgi/konum
    bilgisi olmadığı için ``extract_tables`` boş döner. Bu fallback yalnız
    değişiklik cetveli başlığı bulunan sayfalarda çağrılır.
    """
    rows: list[tuple[str, str, str]] = []
    for line in (page_text or "").splitlines():
        match = _TEXT_TABLE_ROW.match(_compact(line))
        if not match:
            continue
        source = re.sub(r"\s+", "", match.group("source"))
        effective = match.group("effective")
        articles = _compact(match.group("articles"))
        if effective == "-" or articles == "-" or not articles:
            continue
        rows.append((source, effective, articles))
    return rows


def build_amendment_ledger(
    *, output_path: Path | None = None, law_paths: list[Path] | None = None
) -> dict[str, Any]:
    """Kanun sonlarındaki değişiklik cetvellerini tek JSON dosyasına çıkarır."""
    import pdfplumber

    output = output_path or amendment_ledger_config.output_path
    records: list[dict[str, Any]] = []
    files = sorted(law_paths or list(documents_config.laws_path.glob("*.pdf")))
    for path in files:
        previous = ""
        active = False
        with pdfplumber.open(path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                raw_page_text = page.extract_text() or ""
                page_text = fold_turkish(raw_page_text).casefold()
                header_hint = (
                    ("degistiren" in page_text or "degistirilen" in page_text)
                    and "yururluge giris" in page_text
                )
                if not active and not header_hint:
                    continue
                tables = page.extract_tables()
                has_header = any(_is_table(table) for table in tables if table)
                extracted_rows = 0
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    header = _is_table(table)
                    width = max((len(row or []) for row in table), default=0)
                    if not header and not (active and width >= 3):
                        continue
                    active = header or active
                    for row in table[1 if header else 0 :]:
                        cells = [_compact(cell) for cell in (row or [])]
                        if len(cells) < 3:
                            continue
                        source, affected, effective = cells[:3]
                        source = source or previous
                        if not source or not affected:
                            continue
                        previous = source
                        source_no = next(iter(_NUMBER.findall(source)), None)
                        records.append(
                            {
                                "hedef_kanun_no": _law_number(path),
                                "hedef_madde_nolari": _articles(affected),
                                "degistiren_duzenleme_no": source_no,
                                "degistiren_duzenleme_metni": source,
                                "yururluk_tarihi": _date(effective),
                                "yururluk_tarihi_ham": effective,
                                "kaynak_dosya": path.name,
                                "kaynak_sayfa": page_no,
                                "ham_kanit": " | ".join(cells[:3]),
                                "hukuki_etki": _legal_effect(source, affected, effective),
                                "dogrulama_durumu": "pdf_tablosundan_deterministik_cikarim",
                            }
                        )
                        extracted_rows += 1
                # pdfplumber bir sayfanın yalnız bir bölümünü hücrelere
                # ayırabilir. Bu nedenle satır fallback'i her aktif cetvel
                # sayfasında çalışır; aşağıdaki imza aynı kaydın iki kez
                # eklenmesini engeller.
                if active:
                    page_signatures = {
                        (
                            str(row.get("degistiren_duzenleme_no") or ""),
                            tuple(row.get("hedef_madde_nolari") or []),
                            str(row.get("yururluk_tarihi_ham") or ""),
                        )
                        for row in records
                        if row.get("kaynak_dosya") == path.name and row.get("kaynak_sayfa") == page_no
                    }
                    for source, effective, affected in _text_table_rows(raw_page_text):
                        source_no = next(iter(_NUMBER.findall(source)), None)
                        if not source_no:
                            continue
                        signature = (source_no, tuple(_articles(affected)), effective)
                        if signature in page_signatures:
                            continue
                        records.append(
                            {
                                "hedef_kanun_no": _law_number(path),
                                "hedef_madde_nolari": _articles(affected),
                                "degistiren_duzenleme_no": source_no,
                                "degistiren_duzenleme_metni": source,
                                "yururluk_tarihi": _date(effective),
                                "yururluk_tarihi_ham": effective,
                                "kaynak_dosya": path.name,
                                "kaynak_sayfa": page_no,
                                "ham_kanit": f"{source} | {affected} | {effective}",
                                "hukuki_etki": _legal_effect(source, affected, effective),
                                "dogrulama_durumu": "pdf_metin_tablosundan_deterministik_cikarim",
                            }
                        )
                if active and not has_header and not tables:
                    active = False
    payload = {"schema_version": "amendment-ledger-v1", "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"files": len(files), "records": len(records), "path": str(output)}


def is_amendment_query(query: str) -> bool:
    normalized = fold_turkish(query).casefold()
    return any(term in normalized for term in _STRONG_HINTS + _CONTEXT_HINTS)


def lookup_amendment_ledger(query: str, law_number: str | None) -> list[SearchResult]:
    """Hedef kanun ve değiştiren düzenleme numarasıyla kanıt satırlarını bulur."""
    if not amendment_ledger_config.enabled or not is_amendment_query(query):
        return []
    path = amendment_ledger_config.output_path
    if not path.is_file():
        return []
    numbers = {value for value in _NUMBER.findall(query) if not value.startswith(("19", "20"))}
    if law_number:
        numbers.discard(str(law_number))
    if not numbers:
        return []
    records = json.loads(path.read_text(encoding="utf-8")).get("records", [])
    matched = [
        row for row in records
        if (not law_number or str(row.get("hedef_kanun_no")) == str(law_number))
        and str(row.get("degistiren_duzenleme_no") or "") in numbers
    ][: amendment_ledger_config.max_results]
    results: list[SearchResult] = []
    for index, row in enumerate(matched, start=1):
        articles = ", ".join(row.get("hedef_madde_nolari") or []) or "-"
        text = (
            f"Değiştiren düzenleme: {row.get('degistiren_duzenleme_metni')}. "
            f"Etkilenen maddeler: {articles}. "
            f"Yürürlük tarihi: {row.get('yururluk_tarihi_ham') or '-'}"
        )
        results.append(
            SearchResult(
                id=f"ledger:{row.get('hedef_kanun_no')}:{row.get('degistiren_duzenleme_no')}:{index}",
                text=text,
                metadata={
                    "chunk_id": f"ledger:{row.get('hedef_kanun_no')}:{index}",
                    "law_number": str(row.get("hedef_kanun_no") or "unknown"),
                    "article_no": articles,
                    "law_name": f"{row.get('hedef_kanun_no')} sayılı Kanun değişiklik cetveli",
                    "source_file": row.get("kaynak_dosya", "unknown"),
                    "page_start": row.get("kaynak_sayfa"),
                    "page_end": row.get("kaynak_sayfa"),
                    "source_type": "amendment_ledger",
                    "authority_level": "primary_table",
                    "ledger_record": row,
                },
                score=1.0 - index * 0.001,
            )
        )
    return results
