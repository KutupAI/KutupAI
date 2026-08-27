"""Kalıcı ve seçici sohbet hafızası.

Tüm dönüşler SQLite'ta saklanır. Yeni bir soru geldiğinde yalnız aynı
sohbetteki ilgili dönüşler seçilir; geçmişin tamamı modele gönderilmez.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Any, Callable, Iterable, Optional


_LAW_PATTERN = re.compile(r"\b(\d{3,5})\s*(?:sayılı|sayili)\b", re.IGNORECASE)
_AMENDMENT_PATTERN = re.compile(
    r"\b(?:değiştiren\s+düzenleme|degistiren\s+duzenleme|değişiklik\s+cetveli\s+kanıtı)\s*:\s*(\d{3,5})\b",
    re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]{3,}")
_FOLLOW_UP_MARKERS = (
    "peki", "bunun", "buna", "bunda", "onun", "ona", "aynı", "ayni",
    "bu kanun", "bu düzenleme", "bu duzenleme", "hangi madde", "ne zaman",
    "yürürlük", "yururluk", "devamı", "devami",
)


@dataclass(frozen=True)
class MemoryTurn:
    question: str
    answer: str
    laws: tuple[str, ...]
    articles: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class MemoryContext:
    turns: tuple[MemoryTurn, ...] = ()
    focus_law: str = ""
    reference_law: str = ""
    is_follow_up: bool = False

    def for_writer(self, *, max_chars: int = 1800) -> str:
        """Writer'a verilecek kısa, kanıt olmayan konuşma özeti."""
        rows: list[str] = []
        for turn in self.turns:
            laws = f" | Kanun: {', '.join(turn.laws)}" if turn.laws else ""
            rows.append(
                f"Önceki soru: {turn.question[:420]}\n"
                f"Önceki cevap: {turn.answer[:560]}{laws}"
            )
        return "\n\n".join(rows)[:max_chars]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _words(text: str) -> set[str]:
    return {word.casefold() for word in _WORD_PATTERN.findall(text)}


def _laws(text: str) -> set[str]:
    return {match.group(1) for match in _LAW_PATTERN.finditer(text)}


def _vector_to_blob(vector: Iterable[float]) -> bytes:
    values = array("f", (float(value) for value in vector))
    return values.tobytes()


def _blob_to_vector(blob: bytes) -> array:
    values = array("f")
    values.frombytes(blob)
    return values


def _json_object(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _cosine(left: array, right: array) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return float(dot / (left_norm * right_norm)) if left_norm and right_norm else 0.0


class ConversationStore:
    """Sohbet geçmişini ve hafif anlamsal arama indeksini yönetir."""

    def __init__(
        self,
        root: Optional[Path] = None,
        *,
        embedder: Optional[Callable[[str], list[float]]] = None,
        ttl_days: Optional[int] = None,
    ) -> None:
        self.root = root or Path(__file__).resolve().parent / "runtime" / "conversations"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "conversation_memory.sqlite3"
        self._embedder = embedder
        # 0 varsayılanı geçmişi silmez. İstenirse ortam değişkeniyle süreli
        # saklama politikası açılabilir.
        self.ttl_days = ttl_days if ttl_days is not None else int(os.getenv("CONVERSATION_MEMORY_TTL_DAYS", "0"))
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    chat_id TEXT PRIMARY KEY,
                    document_hash TEXT NOT NULL,
                    document_path TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    laws_json TEXT NOT NULL,
                    articles_json TEXT NOT NULL,
                    state_json TEXT NOT NULL DEFAULT '{}',
                    embedding BLOB,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES conversations(chat_id)
                );
                CREATE INDEX IF NOT EXISTS idx_turns_chat_created
                    ON turns(chat_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS ocr_cache (
                    document_hash TEXT PRIMARY KEY,
                    ocr_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            # Önceki denemelerde oluşturulmuş SQLite dosyaları korunur.
            # Yeni hafıza alanları eksikse şema yerinde güncellenir.
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(turns)").fetchall()
            }
            additions = {
                "laws_json": "TEXT NOT NULL DEFAULT '[]'",
                "articles_json": "TEXT NOT NULL DEFAULT '[]'",
                "state_json": "TEXT NOT NULL DEFAULT '{}'",
                "embedding": "BLOB",
            }
            for name, definition in additions.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE turns ADD COLUMN {name} {definition}")

            conversation_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "document_hash" not in conversation_columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN document_hash TEXT")
                # İlk hafıza sürümündeki alan adı document_sha256 idi.
                if "document_sha256" in conversation_columns:
                    connection.execute(
                        "UPDATE conversations SET document_hash = document_sha256 "
                        "WHERE document_hash IS NULL OR document_hash = ''"
                    )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def bind_document(self, chat_id: str, source_path: str) -> str:
        """Yüklenen geçici dosyanın kalıcı bir kopyasını sohbetle bağlar."""
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        document_hash = self._sha256(source)
        safe_chat = hashlib.sha256(chat_id.encode("utf-8")).hexdigest()
        destination_dir = self.root / "files" / safe_chat
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{document_hash[:16]}{source.suffix.lower()}"
        if not destination.exists():
            shutil.copy2(source, destination)

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT document_hash FROM conversations WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            # Aynı sohbet içinde yeni dosya yüklenirse eski bağlam temizlenir.
            if existing and existing["document_hash"] != document_hash:
                connection.execute("DELETE FROM turns WHERE chat_id = ?", (chat_id,))
            column_names = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
            }
            # Eski veritabanında document_sha256 zorunlu kalmış olabilir.
            # Aynı hash'i iki alana yazarak dosyayı silmeden geriye uyum sağlarız.
            if "document_sha256" in column_names:
                connection.execute(
                    """
                    INSERT INTO conversations(
                        chat_id, document_hash, document_sha256, document_path, updated_at
                    ) VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        document_hash=excluded.document_hash,
                        document_sha256=excluded.document_sha256,
                        document_path=excluded.document_path,
                        updated_at=excluded.updated_at
                    """,
                    (chat_id, document_hash, document_hash, str(destination), _now()),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO conversations(chat_id, document_hash, document_path, updated_at)
                    VALUES(?, ?, ?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        document_hash=excluded.document_hash,
                        document_path=excluded.document_path,
                        updated_at=excluded.updated_at
                    """,
                    (chat_id, document_hash, str(destination), _now()),
                )
        return str(destination)

    def ensure_conversation(self, chat_id: str) -> None:
        """Dosya eklenmeden başlayan soru-cevap sohbeti için kayıt oluşturur."""

        with self._connect() as connection:
            column_names = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "document_sha256" in column_names:
                connection.execute(
                    """
                    INSERT INTO conversations(
                        chat_id, document_hash, document_sha256, document_path, updated_at
                    ) VALUES(?, '', '', '', ?)
                    ON CONFLICT(chat_id) DO UPDATE SET updated_at=excluded.updated_at
                    """,
                    (chat_id, _now()),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO conversations(chat_id, document_hash, document_path, updated_at)
                    VALUES(?, '', '', ?)
                    ON CONFLICT(chat_id) DO UPDATE SET updated_at=excluded.updated_at
                    """,
                    (chat_id, _now()),
                )

    def document_path(self, chat_id: str) -> Optional[str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT document_path FROM conversations WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        path = Path(str(row["document_path"])) if row else None
        return str(path) if path and path.is_file() else None

    def document_hash(self, chat_id: str) -> str:
        """Sohbete bağlı dosyanın hash değerini döndürür."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT document_hash FROM conversations WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        return str(row["document_hash"] or "") if row else ""

    def list_conversations(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Sidebar için sohbet başlıklarını en yeniden en eskiye döndürür."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.chat_id, c.updated_at, COUNT(t.id) AS turn_count,
                       COALESCE(
                           (SELECT question FROM turns
                            WHERE chat_id = c.chat_id ORDER BY id ASC LIMIT 1),
                           'Yeni Sohbet'
                       ) AS title
                FROM conversations c
                LEFT JOIN turns t ON t.chat_id = c.chat_id
                GROUP BY c.chat_id, c.updated_at
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [
            {
                "chat_id": str(row["chat_id"]),
                "title": str(row["title"])[:80],
                "updated_at": str(row["updated_at"]),
                "turn_count": int(row["turn_count"]),
            }
            for row in rows
        ]

    def get_conversation(self, chat_id: str) -> Optional[dict[str, Any]]:
        """Bir sohbetin saklanan soru-cevap dönüşlerini verir."""
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT chat_id, updated_at FROM conversations WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            if not conversation:
                return None
            turns = connection.execute(
                """
                SELECT question, answer, state_json, created_at FROM turns
                WHERE chat_id = ? ORDER BY id ASC
                """,
                (chat_id,),
            ).fetchall()
        return {
            "chat_id": str(conversation["chat_id"]),
            "updated_at": str(conversation["updated_at"]),
            "turns": [
                {
                    "question": str(turn["question"]),
                    "answer": str(turn["answer"]),
                    "pipeline_state": _json_object(turn["state_json"]),
                    "created_at": str(turn["created_at"]),
                }
                for turn in turns
            ],
        }

    def delete_conversation(self, chat_id: str) -> bool:
        """Kullanıcının Sidebar'dan sildiği sohbeti ve bağlı dosyasını kaldırır."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT document_path FROM conversations WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            if not row:
                return False
            connection.execute("DELETE FROM turns WHERE chat_id = ?", (chat_id,))
            connection.execute("DELETE FROM conversations WHERE chat_id = ?", (chat_id,))
        path = Path(str(row["document_path"]))
        try:
            if path.is_file():
                path.unlink()
            parent = path.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            # Veritabanı kaydı silindikten sonra dosya silinemese bile sohbet geri gelmez.
            pass
        return True

    def get_ocr_cache(self, document_hash: str) -> Optional[Dict[str, Any]]:
        """Geçerli ise daha önce üretilen birleşik OCR çıktısını döndürür."""
        if not document_hash:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT ocr_json FROM ocr_cache WHERE document_hash = ?", (document_hash,)
            ).fetchone()
        if not row:
            return None
        try:
            value = json.loads(str(row["ocr_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        ocr_data = value.get("ocr_data") if isinstance(value, dict) else None
        if not isinstance(ocr_data, dict) or not str(ocr_data.get("full_text") or "").strip():
            return None
        return value

    def save_ocr_cache(self, document_hash: str, ocr: Any) -> None:
        """Başarılı OCR çıktısını dosya hash'i ile kalıcı olarak saklar."""
        if not document_hash or not isinstance(ocr, dict):
            return
        ocr_data = ocr.get("ocr_data") if isinstance(ocr.get("ocr_data"), dict) else None
        if not ocr.get("success") or not isinstance(ocr_data, dict):
            return
        if not str(ocr_data.get("full_text") or "").strip():
            return
        payload = json.dumps(ocr, ensure_ascii=False, separators=(",", ":"))
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ocr_cache(document_hash, ocr_json, created_at, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(document_hash) DO UPDATE SET
                    ocr_json=excluded.ocr_json,
                    updated_at=excluded.updated_at
                """,
                (document_hash, payload, now, now),
            )

    def _embed(self, text: str) -> Optional[array]:
        try:
            if self._embedder is None:
                from RAG.embeddings.embedding_model import embed_text

                self._embedder = embed_text
            return array("f", self._embedder(text))
        except Exception:
            # Hafıza araması çalışmasa dahi ana workflow durmaz.
            return None

    def search(self, chat_id: str, question: str, *, limit: int = 3) -> MemoryContext:
        """Yeni soruyla ilgili eski dönüşleri döndürür; zayıf eşleşmeleri eler."""
        question = " ".join(question.split())
        if not question:
            return MemoryContext()
        question_laws = _laws(question)
        question_words = _words(question)
        is_marker_follow_up = any(marker in question.casefold() for marker in _FOLLOW_UP_MARKERS)
        query_vector = self._embed(question)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT question, answer, laws_json, articles_json, embedding, created_at
                FROM turns WHERE chat_id = ? ORDER BY id DESC LIMIT 250
                """,
                (chat_id,),
            ).fetchall()

        ranked: list[MemoryTurn] = []
        for rank, row in enumerate(rows):
            laws = tuple(json.loads(row["laws_json"]))
            # Açık yeni kanun, bağlamı kapatır. Ancak “peki/bu düzenleme” gibi
            # takip sorularında sayı çoğu kez sadece karşılaştırma referansıdır.
            if question_laws and laws and not question_laws.intersection(laws) and not is_marker_follow_up:
                continue
            candidate_text = f"{row['question']} {row['answer']}"
            candidate_words = _words(candidate_text)
            overlap = len(question_words & candidate_words) / max(1, len(question_words))
            semantic = 0.0
            if query_vector is not None and row["embedding"]:
                semantic = max(0.0, _cosine(query_vector, _blob_to_vector(row["embedding"])))
            law_bonus = 0.12 if question_laws and question_laws.intersection(laws) else 0.0
            follow_up_bonus = 0.20 if is_marker_follow_up else 0.0
            recency_bonus = max(0.0, 0.04 - rank * 0.002)
            score = 0.78 * semantic + 0.18 * overlap + law_bonus + follow_up_bonus + recency_bonus
            ranked.append(
                MemoryTurn(
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    laws=laws,
                    articles=tuple(json.loads(row["articles_json"])),
                    score=score,
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        threshold = 0.24 if is_marker_follow_up else 0.46 if question_laws else 0.62
        selected = tuple(item for item in ranked[:limit] if item.score >= threshold)
        if not selected:
            return MemoryContext()
        law_candidates = [law for item in selected for law in item.laws]
        focus_law = next((law for law in law_candidates if law), "")
        # Son söylenen farklı kanun, “bu düzenleme” gibi zamirler için
        # değişiklik referansı olarak kullanılır (hedef kanunla karıştırılmaz).
        reference_law = next(
            (law for law in reversed(law_candidates) if law and law != focus_law and law not in question_laws),
            "",
        )
        return MemoryContext(
            turns=selected,
            focus_law=focus_law,
            reference_law=reference_law,
            is_follow_up=True,
        )

    def record_turn(
        self,
        chat_id: str,
        question: str,
        answer: str,
        rag: Any,
        pipeline_state: Optional[dict[str, Any]] = None,
    ) -> None:
        """Tamamlanan dönüşü bir sonraki sorular için indeksler."""
        question, answer = " ".join(question.split()), " ".join(answer.split())
        if not question or not answer:
            return
        results = ((rag or {}).get("rag_data") or {}).get("results") if isinstance(rag, dict) else []
        laws: list[str] = []
        articles: list[str] = []

        def add_law(value: str) -> None:
            if value and value not in laws:
                laws.append(value)

        if isinstance(results, list):
            for result in results[:5]:
                if not isinstance(result, dict):
                    continue
                law = str(result.get("law_number") or "").strip()
                article = str(result.get("article_no") or "").strip()
                add_law(law)
                for amendment in _AMENDMENT_PATTERN.findall(str(result.get("text") or "")):
                    add_law(amendment)
                if article and article not in articles:
                    articles.append(article)
        for mentioned_law in _LAW_PATTERN.findall(f"{question}\n{answer}"):
            add_law(mentioned_law)
        vector = self._embed(f"{question}\n{answer[:700]}")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO turns(
                    chat_id, question, answer, laws_json, articles_json, state_json, embedding, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    question[:1200],
                    answer[:900],
                    json.dumps(laws, ensure_ascii=False),
                    json.dumps(articles, ensure_ascii=False),
                    json.dumps(pipeline_state or {}, ensure_ascii=False, separators=(",", ":")),
                    _vector_to_blob(vector) if vector is not None else None,
                    _now(),
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE chat_id = ?", (_now(), chat_id)
            )
            if self.ttl_days > 0:
                expiry = (datetime.now(UTC) - timedelta(days=self.ttl_days)).isoformat(timespec="seconds")
                connection.execute("DELETE FROM turns WHERE created_at < ?", (expiry,))
