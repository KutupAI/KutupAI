"""Yerel hukukî metadata, ilişki ve FTS indeksi.

Chroma yalnız anlam vektörlerini tutar. Bu modül aynı corpus'un doğrulanabilir
yapılandırılmış görünümünü SQLite'ta üretir; ek servis veya ağ bağlantısı gerekmez.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Sequence

from RAG.configuration.rag_config_loader import legal_index_config
from RAG.retriever.text_utils import fold_turkish, tokenize
from RAG.vector_store.vector_store_interface import SearchResult


_LAW_REFERENCE = re.compile(r"\b(\d{3,5})\s+say[ıi]l[ıi]\s+Kanun\b", re.IGNORECASE)
_ARTICLE_REFERENCE = re.compile(
    r"\b(\d{3,5})\s+say[ıi]l[ıi]\s+Kanun\b.{0,180}?"
    r"(?:Madde\s*(\d+[A-Za-z]?)|(\d+[A-Za-z]?)\s*\.\s*madd)",
    re.IGNORECASE | re.DOTALL,
)
_IDENTIFIER = re.compile(r"\b(?:KHK\s*[-/]?\s*)?\d{2,5}(?:/\d+)?\b", re.IGNORECASE)
_catalog_cache: tuple[int, dict[str, set[str]]] | None = None


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _as_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [str(value)]


def _fts_query(query: str) -> str:
    """FTS sorgusunu yalnız güvenli token'lardan oluşturur."""
    terms = tokenize(fold_turkish(query).casefold(), min_len=2)
    unique = list(dict.fromkeys(term for term in terms if term))[:16]
    return " OR ".join(f'"{term}"' for term in unique)


def _fts_all_terms_query(query: str) -> str:
    """Bir hukuk adının bütün anlamlı kelimelerini birlikte arar."""
    terms = tokenize(fold_turkish(query).casefold(), min_len=2)
    unique = list(dict.fromkeys(term for term in terms if term and term != "kanunu"))[:5]
    return " AND ".join(f'"{term}"' for term in unique)


def _natural_article(value: object) -> tuple[int, str]:
    match = re.search(r"\d+", str(value or ""))
    return (int(match.group()) if match else 10**9, str(value or ""))


class LegalIndex:
    """SQLite dosyasının kurulum, structured lookup ve FTS işlemleri."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or legal_index_config.output_path

    def available(self) -> bool:
        return legal_index_config.enabled and self.path.is_file()

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            return sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.path)

    @staticmethod
    def _schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE documents (
                source_file TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                document_category TEXT,
                law_number TEXT,
                law_name TEXT,
                acceptance_date TEXT,
                publication_date TEXT,
                effective_date TEXT
            );

            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                source_type TEXT NOT NULL,
                document_category TEXT,
                law_number TEXT,
                law_name TEXT,
                article_no TEXT,
                article_type TEXT,
                article_title TEXT,
                structural_path TEXT,
                paragraph_no TEXT,
                clause_no TEXT,
                legal_status TEXT,
                page_start INTEGER,
                page_end INTEGER,
                full_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX chunks_law_article_idx ON chunks(law_number, article_no);
            CREATE INDEX chunks_source_idx ON chunks(source_file);

            CREATE VIRTUAL TABLE chunk_fts USING fts5(
                chunk_id UNINDEXED,
                source_file UNINDEXED,
                law_number UNINDEXED,
                article_no UNINDEXED,
                full_text,
                tokenize='unicode61 remove_diacritics 2'
            );

            CREATE TABLE legal_facts (
                fact_id TEXT PRIMARY KEY,
                fact_type TEXT NOT NULL,
                source_chunk_id TEXT,
                source_file TEXT,
                law_number TEXT,
                article_no TEXT,
                values_json TEXT NOT NULL,
                evidence_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                extraction_method TEXT NOT NULL
            );
            CREATE INDEX legal_facts_type_law_idx ON legal_facts(fact_type, law_number);

            CREATE TABLE amendments (
                record_id TEXT PRIMARY KEY,
                target_law_number TEXT NOT NULL,
                amending_number TEXT,
                amending_text TEXT,
                affected_articles TEXT,
                legal_effect TEXT,
                effective_date TEXT,
                effective_date_raw TEXT,
                source_file TEXT,
                source_page INTEGER,
                evidence_text TEXT NOT NULL,
                verification_status TEXT NOT NULL
            );
            CREATE INDEX amendments_lookup_idx ON amendments(target_law_number, amending_number);

            CREATE TABLE legal_edges (
                edge_id TEXT PRIMARY KEY,
                source_chunk_id TEXT,
                source_law_number TEXT,
                source_article_no TEXT,
                target_law_number TEXT,
                target_article_no TEXT,
                relation_type TEXT NOT NULL,
                evidence_text TEXT NOT NULL
            );
            CREATE INDEX legal_edges_source_idx ON legal_edges(source_law_number, source_article_no);
            """
        )

    def rebuild(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        facts: Sequence[dict[str, Any]] = (),
        amendments: Sequence[dict[str, Any]] = (),
    ) -> dict[str, int | str]:
        """Tek bir atomik dosyada corpus'un structured görünümünü üretir."""
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        documents: dict[str, tuple[Any, ...]] = {}
        chunks: list[tuple[Any, ...]] = []
        fts_rows: list[tuple[Any, ...]] = []
        edges: list[tuple[Any, ...]] = []
        article_nodes: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for row in rows:
            metadata = dict(row.get("metadata") or {})
            text = str(row.get("text") or "").strip()
            chunk_id = str(row.get("chunk_id") or metadata.get("chunk_id") or "").strip()
            if not chunk_id or not text:
                continue
            source_file = str(metadata.get("source_file") or "unknown")
            source_type = str(metadata.get("source_type") or "unknown")
            law_number = str(metadata.get("law_number") or "unknown")
            article_no = str(metadata.get("article_no") or metadata.get("article_number") or "unknown")
            page_start = metadata.get("page_start") or metadata.get("page")
            page_end = metadata.get("page_end") or metadata.get("page")
            documents[source_file] = (
                source_file, source_type, str(metadata.get("document_category") or "unknown"), law_number,
                str(metadata.get("law_name") or Path(source_file).stem),
                str(metadata.get("acceptance_date") or ""), str(metadata.get("publication_date") or ""),
                str(metadata.get("effective_date") or ""),
            )
            chunks.append((
                chunk_id, source_file, source_type, str(metadata.get("document_category") or "unknown"),
                law_number, str(metadata.get("law_name") or Path(source_file).stem), article_no,
                str(metadata.get("article_type") or "unknown"), str(metadata.get("article_title") or ""),
                str(metadata.get("structural_path") or ""), str(metadata.get("paragraph_no") or ""),
                str(metadata.get("clause_no") or ""), str(metadata.get("legal_status") or "consolidated"),
                int(page_start) if str(page_start or "").isdigit() else None,
                int(page_end) if str(page_end or "").isdigit() else None,
                text, _json(metadata),
            ))
            # FTS tablosu Türkçe karakterlerden arındırılmış eşdeğer metni
            # saklar; sonuç ise her zaman chunks.full_text içinden döner.
            fts_rows.append((chunk_id, source_file, law_number, article_no, fold_turkish(text).casefold()))
            if law_number != "unknown" and article_no != "unknown":
                article_nodes[law_number].append((article_no, chunk_id))
            for target_law, target_article_a, target_article_b in _ARTICLE_REFERENCE.findall(text):
                target_article = target_article_a or target_article_b or ""
                edge_id = f"reference:{chunk_id}:{target_law}:{target_article}"
                edges.append((edge_id, chunk_id, law_number, article_no, target_law, target_article, "cross_reference", text[:800]))
            for target_law in sorted(set(_LAW_REFERENCE.findall(text))):
                edge_id = f"law-reference:{chunk_id}:{target_law}"
                edges.append((edge_id, chunk_id, law_number, article_no, target_law, "", "law_reference", text[:800]))

        for law_number, nodes in article_nodes.items():
            ordered = sorted({node for node in nodes}, key=lambda item: _natural_article(item[0]))
            for (article_a, chunk_a), (article_b, _chunk_b) in zip(ordered, ordered[1:]):
                edge_id = f"adjacent:{law_number}:{article_a}:{article_b}"
                edges.append((edge_id, chunk_a, law_number, article_a, law_number, article_b, "adjacent_article", "Aynı kanunda komşu maddeler."))

        fact_rows: list[tuple[Any, ...]] = []
        for fact in facts:
            evidence = dict(fact.get("evidence") or {})
            fact_rows.append((
                str(fact.get("fact_id") or ""), str(fact.get("fact_type") or "unknown"),
                str(evidence.get("chunk_id") or ""), str(evidence.get("source_file") or "unknown"),
                str(evidence.get("law_number") or "unknown"), str(evidence.get("article_no") or "unknown"),
                _json(fact.get("values") or {}), str(fact.get("evidence_text") or ""),
                float(fact.get("confidence") or 0.0), str(fact.get("extraction_method") or "unknown"),
            ))

        amendment_rows: list[tuple[Any, ...]] = []
        for index, record in enumerate(amendments, start=1):
            target_law = str(record.get("hedef_kanun_no") or "unknown")
            amending = str(record.get("degistiren_duzenleme_no") or "")
            affected = ", ".join(_as_list(record.get("hedef_madde_nolari")))
            evidence_text = str(record.get("ham_kanit") or "")
            record_id = f"amendment:{target_law}:{amending}:{affected}:{index}"
            amendment_rows.append((
                record_id, target_law, amending, str(record.get("degistiren_duzenleme_metni") or ""), affected,
                str(record.get("hukuki_etki") or "degisiklik_veya_iptal_cetvel_kaydi"),
                str(record.get("yururluk_tarihi") or ""), str(record.get("yururluk_tarihi_ham") or ""),
                str(record.get("kaynak_dosya") or "unknown"),
                int(record.get("kaynak_sayfa")) if str(record.get("kaynak_sayfa") or "").isdigit() else None,
                evidence_text, str(record.get("dogrulama_durumu") or "automatic"),
            ))

        with closing(sqlite3.connect(temporary)) as connection:
            self._schema(connection)
            connection.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)", documents.values())
            connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", chunks)
            connection.executemany("INSERT INTO chunk_fts VALUES (?, ?, ?, ?, ?)", fts_rows)
            connection.executemany("INSERT OR IGNORE INTO legal_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", edges)
            connection.executemany("INSERT OR REPLACE INTO legal_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", fact_rows)
            connection.executemany("INSERT OR REPLACE INTO amendments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", amendment_rows)
            connection.commit()
        temporary.replace(self.path)
        return {
            "path": str(self.path), "documents": len(documents), "chunks": len(chunks),
            "facts": len(fact_rows), "amendments": len(amendment_rows), "edges": len({row[0] for row in edges}),
        }

    def law_catalog(self) -> dict[str, set[str]]:
        """Dosya ve kanun adlarından bir kez okunabilen dinamik başlık kataloğu."""
        global _catalog_cache
        if not self.available():
            return {}
        modified = self.path.stat().st_mtime_ns
        if _catalog_cache and _catalog_cache[0] == modified:
            return _catalog_cache[1]
        with closing(self._connect(readonly=True)) as connection:
            rows = connection.execute("SELECT law_number, law_name, source_file FROM documents WHERE law_number != 'unknown'").fetchall()
        catalog: dict[str, set[str]] = defaultdict(set)
        for law_number, law_name, source_file in rows:
            title = fold_turkish(f"{law_name} {Path(str(source_file)).stem}").casefold()
            tokens = tokenize(title, min_len=3)
            catalog[str(law_number)].update(tokens)
            # OHAL gibi dosya başlığından türetilebilen kısa yazımlar için
            # sabit sözlük gerekmez: ilk sözcüğün baş harfi + kısa komşu sözcük.
            words = [word for word in re.findall(r"[a-z0-9]+", title) if not word.isdigit()]
            for left, right in zip(words, words[1:]):
                if len(right) <= 5:
                    catalog[str(law_number)].add(left[:1] + right)
        result = dict(catalog)
        _catalog_cache = (modified, result)
        return result

    def fts_search(self, query: str, *, top_k: int, law_numbers: Sequence[str] = ()) -> list[SearchResult]:
        if not self.available() or not _fts_query(query):
            return []
        sql = (
            "SELECT c.chunk_id, c.full_text, c.metadata_json, bm25(chunk_fts) AS rank "
            "FROM chunk_fts JOIN chunks c ON c.chunk_id = chunk_fts.chunk_id "
            "WHERE chunk_fts MATCH ?"
        )
        params: list[Any] = [_fts_query(query)]
        if law_numbers:
            placeholders = ",".join("?" for _ in law_numbers)
            sql += f" AND c.law_number IN ({placeholders})"
            params.extend(str(value) for value in law_numbers)
        sql += " ORDER BY rank LIMIT ?"
        params.append(max(1, top_k))
        try:
            with closing(self._connect(readonly=True)) as connection:
                rows = connection.execute(sql, params).fetchall()
        except sqlite3.Error:
            return []
        output: list[SearchResult] = []
        for index, (chunk_id, text, metadata_json, rank) in enumerate(rows, start=1):
            metadata = json.loads(metadata_json)
            metadata.update({"structured_fts": True, "fts_rank": index})
            output.append(SearchResult(id=str(chunk_id), text=str(text), metadata=metadata, score=round(max(0.1, 0.96 - index * 0.02), 4)))
        return output

    def fts_all_terms_search(self, query: str, *, top_k: int) -> list[SearchResult]:
        """Başlık/atıf aramasında OR yerine bütün terimleri zorunlu tutar."""
        match_query = _fts_all_terms_query(query)
        if not self.available() or not match_query:
            return []
        sql = (
            "SELECT c.chunk_id, c.full_text, c.metadata_json, bm25(chunk_fts) AS rank "
            "FROM chunk_fts JOIN chunks c ON c.chunk_id = chunk_fts.chunk_id "
            "WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?"
        )
        try:
            with closing(self._connect(readonly=True)) as connection:
                rows = connection.execute(sql, (match_query, max(1, top_k))).fetchall()
        except sqlite3.Error:
            return []
        output: list[SearchResult] = []
        for index, (chunk_id, text, metadata_json, _rank) in enumerate(rows, start=1):
            metadata = json.loads(metadata_json)
            metadata.update({"structured_fts": True, "fts_all_terms": True, "fts_rank": index})
            output.append(SearchResult(
                id=str(chunk_id), text=str(text), metadata=metadata,
                score=round(max(0.1, 0.98 - index * 0.02), 4),
            ))
        return output

    def relation_edges(self) -> list[tuple[str, str, str, str, str]]:
        """Graph-RAG için yalnız extractor tarafından kanıtlanmış ilişkileri verir."""
        if not self.available():
            return []
        try:
            with closing(self._connect(readonly=True)) as connection:
                rows = connection.execute(
                    "SELECT source_law_number, source_article_no, target_law_number, target_article_no, relation_type "
                    "FROM legal_edges"
                ).fetchall()
            return [tuple(str(value or "") for value in row) for row in rows]
        except sqlite3.Error:
            return []

    def structured_lookup(self, query: str, frame: Any, *, top_k: int = 8) -> list[SearchResult]:
        """Tablo/olgu kaydını mümkünse her zaman ham kanıt chunk'ına çözer."""
        if not self.available():
            return []
        normalized = fold_turkish(query).casefold()
        identifiers = {
            re.sub(r"^KHK\s*[-/]?\s*", "", value, flags=re.IGNORECASE).strip()
            for value in _IDENTIFIER.findall(query)
        }
        # Tahmin edilen kanun adı, tablo sorgusunu tek başına kapatmaz. Yalnız
        # kullanıcıdaki açık atıf (strict_law_numbers) SQL hedef filtresidir.
        # Bir karşılaştırma sorusu iki açık kanun başlığı taşıyabilir. Yalnız
        # ilk strict filtreyi kullanmak diğer kanunun amendment kaydını SQL'de
        # daha arama başlamadan dışarıda bırakır.
        target_laws = getattr(frame, "target_law_numbers", ())
        laws = tuple(str(value) for value in target_laws if str(value) and str(value) != "unknown") or tuple(
            str(value) for value in getattr(frame, "strict_law_numbers", ()) if str(value) and str(value) != "unknown"
        )
        amendment_numbers = tuple(str(value) for value in getattr(frame, "amending_numbers", ()) if str(value))
        wants_history = bool(getattr(frame, "needs_amendment_evidence", False))
        # Yasama işlemindeki "iptal" veya sıradan "karar" kelimesi tek
        # başına mahkeme kararı değildir. Court facts yalnız açık yargı sinyaliyle
        # getirilir; aksi hâlde değişiklik sorularının aday havuzu kirlenir.
        wants_court = any(term in normalized for term in (
            "anayasa mahkem", "yuksek mahk", "mahkeme karari", "esas say", "karar say",
        ))
        wants_duration = bool(re.search(r"\b(kac|hangi|en gec).{0,24}\b(saat|gun|ay|yil)\b", normalized))
        output: list[SearchResult] = []
        seen: set[str] = set()
        try:
            with closing(self._connect(readonly=True)) as connection:
                # Düzenleme numarası yoksa bütün değişiklik cetvelini Top-K'ye
                # sokmak yanlış tarihi/maddeyi öne çıkarır. Bu durumda hybrid
                # ham hüküm üzerinden çalışır; cetvel yalnız açık düzenleme/KHK
                # numarası bulunduğunda kesin kanıt olur.
                if wants_history and amendment_numbers:
                    conditions: list[str] = []
                    params: list[Any] = []
                    if laws:
                        conditions.append("target_law_number IN (%s)" % ",".join("?" for _ in laws))
                        params.extend(laws)
                    conditions.append("amending_number IN (%s)" % ",".join("?" for _ in amendment_numbers))
                    params.extend(amendment_numbers)
                    rows = connection.execute(
                        "SELECT record_id, target_law_number, amending_number, amending_text, affected_articles, legal_effect, "
                        "effective_date_raw, source_file, source_page, evidence_text, verification_status "
                        "FROM amendments WHERE " + " AND ".join(conditions) + " LIMIT ?",
                        [*params, max(1, top_k)],
                    ).fetchall()
                    # Bir düzenleme cetvelde birden çok satırla yer alabilir.
                    # Aynı hedef kanun+düzenleme satırlarını tek kanıt sonucu
                    # yaparak tarih/madde varyantlarının kaybolmasını önleriz.
                    grouped_rows: dict[tuple[str, str, str], list[tuple[Any, ...]]] = defaultdict(list)
                    for record in rows:
                        grouped_rows[(str(record[1]), str(record[2]), str(record[7]))].append(record)
                    rows = []
                    for (target, amending, source_file), records in grouped_rows.items():
                        first = records[0]
                        rows.append((
                            f"amendment-group:{target}:{amending}:{first[0]}", target, amending, first[3],
                            " | ".join(dict.fromkeys(str(record[4] or "-") for record in records)), first[5],
                            " | ".join(dict.fromkeys(str(record[6] or "-") for record in records)), source_file, first[8],
                            "\n".join(str(record[9]) for record in records), first[10],
                        ))
                    for row in rows:
                        record_id, target, amending, amending_text, articles, effect, date_raw, source_file, page, evidence, verification = row
                        text = (
                            f"Değişiklik cetveli kanıtı: {evidence}. Değiştiren düzenleme: {amending_text}. "
                            f"Etkilenen maddeler: {articles or '-'}. Yürürlük tarihi: {date_raw or '-'}."
                        )
                        metadata = {
                            "chunk_id": record_id, "law_number": target, "law_name": f"{target} sayılı Kanun değişiklik cetveli",
                            "article_no": articles or "unknown", "source_file": source_file, "source_type": "amendment_ledger",
                            "page_start": page, "page_end": page, "legal_effect": effect,
                            "verification_status": verification, "evidence_slot": "amendment",
                            "amending_number": str(amending or ""),
                        }
                        output.append(SearchResult(id=record_id, text=text, metadata=metadata, score=0.995))
                        seen.add(record_id)

                fact_types: list[str] = []
                if wants_court:
                    fact_types.append("constitutional_court_annulment")
                if wants_duration:
                    fact_types.append("legal_time_limit")
                if fact_types:
                    type_markers = ",".join("?" for _ in fact_types)
                    sql = (
                        "SELECT fact_id, fact_type, source_chunk_id, source_file, law_number, article_no, values_json, evidence_text "
                        f"FROM legal_facts WHERE fact_type IN ({type_markers})"
                    )
                    params = list(fact_types)
                    if laws:
                        sql += " AND law_number IN (%s)" % ",".join("?" for _ in laws)
                        params.extend(laws)
                    # Aynı soruda iki mahkeme kararı bulunabilir. Her satırda
                    # bütün E./K. numaralarını istemek iki doğru kaydı da eler;
                    # bu nedenle ayırt edici karar numaraları OR ile aranır.
                    decision_identifiers = sorted(
                        value for value in identifiers if "/" in value and len(value) >= 5
                    )
                    if decision_identifiers:
                        alternatives = " OR ".join(
                            "(values_json LIKE ? OR evidence_text LIKE ?)" for _ in decision_identifiers
                        )
                        sql += f" AND ({alternatives})"
                        for value in decision_identifiers:
                            params.extend([f"%{value}%", f"%{value}%"])
                    article_numbers = tuple(str(value) for value in getattr(frame, "article_numbers", ()) if str(value))
                    if wants_court and article_numbers:
                        sql += " AND article_no IN (%s)" % ",".join("?" for _ in article_numbers)
                        params.extend(article_numbers)
                    sql += " LIMIT ?"
                    params.append(max(1, top_k * 3))
                    fact_rows = connection.execute(sql, params).fetchall()
                    for fact_id, fact_type, source_chunk_id, source_file, law_number, article_no, values_json, evidence in fact_rows:
                        candidate_text = f"{values_json} {evidence}".casefold()
                        if decision_identifiers and not any(value.casefold() in candidate_text for value in decision_identifiers):
                            continue
                        raw = connection.execute(
                            "SELECT full_text, metadata_json FROM chunks WHERE chunk_id = ?", (source_chunk_id,)
                        ).fetchone()
                        if raw:
                            text, metadata_json = raw
                            metadata = json.loads(metadata_json)
                        else:
                            text = f"Hukukî olgu: {values_json}. Kanıt: {evidence}"
                            metadata = {}
                        metadata.update({
                            "chunk_id": source_chunk_id or f"fact:{fact_id}", "source_file": source_file,
                            "law_number": law_number, "article_no": article_no, "structured_fact_type": fact_type,
                            "evidence_slot": "court" if fact_type == "constitutional_court_annulment" else "duration",
                        })
                        result_id = str(source_chunk_id or f"fact:{fact_id}")
                        if result_id not in seen:
                            output.append(SearchResult(id=result_id, text=str(text), metadata=metadata, score=0.992))
                            seen.add(result_id)
        except (sqlite3.Error, json.JSONDecodeError):
            return output[:top_k]
        return output[:top_k]


_instance: LegalIndex | None = None


def get_legal_index() -> LegalIndex:
    global _instance
    if _instance is None:
        _instance = LegalIndex()
    return _instance


def rebuild_legal_index(rows: Iterable[dict[str, Any]], *, facts_path: Path, ledger_path: Path) -> dict[str, int | str]:
    """JSON denetim kayıtlarını okuyup yerel SQLite indeksi üretir."""
    facts_payload = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.is_file() else {}
    ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {}
    return get_legal_index().rebuild(
        rows,
        facts=list(facts_payload.get("records") or []),
        amendments=list(ledger_payload.get("records") or []),
    )
