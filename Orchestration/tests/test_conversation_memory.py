from pathlib import Path
import sqlite3

from Orchestration.conversation_store import ConversationStore
from Orchestration.graph.graph_definition import Stage
from Orchestration.process_service import run_workflow
from Orchestration.tests.mock_agents import (
    MockClassificationAgent,
    MockExtractionAgent,
    MockOCRAgent,
    MockRagAgent,
    MockRoutingAgent,
    MockSummaryAgent,
    MockValidationAgent,
    MockWriterAgent,
)


def _embed(text: str) -> list[float]:
    normalized = text.casefold()
    return [
        1.0 if "4483" in normalized else 0.0,
        1.0 if "4734" in normalized else 0.0,
        1.0 if any(word in normalized for word in ("yürürlük", "yururluk", "tarih")) else 0.0,
        1.0 if any(word in normalized for word in ("madde", "değişiklik", "degisiklik")) else 0.0,
    ]


def test_memory_selects_related_turn_but_rejects_a_new_law(tmp_path: Path):
    store = ConversationStore(tmp_path / "memory", embedder=_embed)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"test document")
    store.bind_document("chat-1", str(source))
    store.record_turn(
        "chat-1",
        "4483 sayılı Kanunda hangi değişiklik yapıldı?",
        "7196 sayılı Kanun 3. maddeyi değiştirdi; yürürlük tarihi 24/12/2019.",
        {"rag_data": {"results": [{"law_number": "4483", "article_no": "3"}]}},
    )

    follow_up = store.search("chat-1", "Peki yürürlük tarihi nedir?")
    assert follow_up.is_follow_up is True
    assert follow_up.focus_law == "4483"
    assert len(follow_up.turns) == 1

    new_topic = store.search("chat-1", "4734 sayılı Kanunda itiraz süresi nedir?")
    assert new_topic.is_follow_up is False
    assert new_topic.turns == ()


def test_follow_up_keeps_previous_amendment_when_current_law_is_comparison_base(tmp_path: Path):
    store = ConversationStore(tmp_path / "memory", embedder=_embed)
    store.record_turn(
        "chat-amendment",
        "4483 sayılı Kanunda 7547 sayılı Kanunun etkilediği maddeler nelerdir?",
        "7547 sayılı Kanun 3, 12, 13 ve Geçici Madde 4'ü etkiledi.",
        {
            "rag_data": {
                "results": [
                    {
                        "law_number": "4483",
                        "article_no": "3, 12, 13, Geçici Madde 4",
                        "text": "Değiştiren düzenleme: 7547. Yürürlük tarihi: 16/5/2025.",
                    }
                ]
            }
        },
    )

    context = store.search(
        "chat-amendment",
        "Peki, bu düzenleme 7196 sayılı Kanundan kaç yıl sonra yürürlüğe girmiştir?",
    )

    assert context.is_follow_up is True
    assert context.focus_law == "4483"
    assert context.reference_law == "7547"


def test_memory_migrates_a_previous_sqlite_schema(tmp_path: Path):
    root = tmp_path / "memory"
    root.mkdir()
    connection = sqlite3.connect(root / "conversation_memory.sqlite3")
    connection.executescript(
        """
        CREATE TABLE conversations (
            chat_id TEXT PRIMARY KEY,
            document_hash TEXT NOT NULL,
            document_path TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    connection.close()

    ConversationStore(root, embedder=_embed)
    connection = sqlite3.connect(root / "conversation_memory.sqlite3")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(turns)")}
    connection.close()
    assert {"laws_json", "articles_json", "embedding"}.issubset(columns)


def test_memory_migrates_the_old_document_sha256_column(tmp_path: Path):
    root = tmp_path / "memory"
    root.mkdir()
    connection = sqlite3.connect(root / "conversation_memory.sqlite3")
    connection.executescript(
        """
        CREATE TABLE conversations (
            chat_id TEXT PRIMARY KEY,
            document_path TEXT NOT NULL,
            document_sha256 TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO conversations VALUES (?, ?, ?, ?)",
        ("chat-old", "C:/old.pdf", "old-hash", "2026-08-01T00:00:00+00:00"),
    )
    connection.commit()
    connection.close()

    ConversationStore(root, embedder=_embed)
    connection = sqlite3.connect(root / "conversation_memory.sqlite3")
    stored_hash = connection.execute(
        "SELECT document_hash FROM conversations WHERE chat_id = 'chat-old'"
    ).fetchone()[0]
    connection.close()
    assert stored_hash == "old-hash"


def test_memory_saves_new_documents_with_legacy_required_hash_column(tmp_path: Path):
    root = tmp_path / "memory"
    root.mkdir()
    connection = sqlite3.connect(root / "conversation_memory.sqlite3")
    connection.executescript(
        """
        CREATE TABLE conversations (
            chat_id TEXT PRIMARY KEY,
            document_path TEXT NOT NULL,
            document_sha256 TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    connection.close()
    source = tmp_path / "source.pdf"
    source.write_bytes(b"test document")

    store = ConversationStore(root, embedder=_embed)
    store.bind_document("chat-new", str(source))

    connection = sqlite3.connect(root / "conversation_memory.sqlite3")
    row = connection.execute(
        "SELECT document_hash, document_sha256 FROM conversations WHERE chat_id = 'chat-new'"
    ).fetchone()
    connection.close()
    assert row[0] == row[1]


def test_follow_up_uses_persisted_document_after_temp_file_is_removed(tmp_path: Path):
    store = ConversationStore(tmp_path / "memory", embedder=_embed)
    temporary_upload = tmp_path / "upload.pdf"
    temporary_upload.write_bytes(b"%PDF-1.4 fake")
    ocr_agent = MockOCRAgent()
    overrides = {
        Stage.OCR: ocr_agent,
        Stage.CLASSIFICATION: MockClassificationAgent(),
        Stage.EXTRACTION: MockExtractionAgent(),
        Stage.VALIDATION: MockValidationAgent(),
        Stage.RAG: MockRagAgent(),
        Stage.SUMMARY: MockSummaryAgent(),
        Stage.ROUTING: MockRoutingAgent(),
        Stage.WRITING: MockWriterAgent(),
    }

    first = run_workflow(
        document_id="chat-persist",
        document_path=str(temporary_upload),
        accompanying_text="4483 sayılı Kanunda hangi madde değişti?",
        agent_overrides=overrides,
        conversation_store=store,
    )
    assert first["Success"] is True
    temporary_upload.unlink()

    second = run_workflow(
        document_id="chat-persist",
        document_path=None,
        accompanying_text="Peki yürürlük tarihi nedir?",
        agent_overrides=overrides,
        conversation_store=store,
    )
    assert second["Success"] is True
    assert store.document_path("chat-persist") is not None
    assert store.get_ocr_cache(store.document_hash("chat-persist")) is not None
    assert ocr_agent.calls == 1


def test_sidebar_history_lists_reads_and_deletes_conversations(tmp_path: Path):
    store = ConversationStore(tmp_path / "memory", embedder=_embed)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"test document")
    store.bind_document("chat-sidebar", str(source))
    state = {"summary": {"success": True, "rag_summary_text": "Detay metni"}}
    store.record_turn("chat-sidebar", "İlk soru nedir?", "İlk cevap budur.", {}, state)

    items = store.list_conversations()
    assert items[0]["chat_id"] == "chat-sidebar"
    assert items[0]["title"] == "İlk soru nedir?"

    detail = store.get_conversation("chat-sidebar")
    assert detail is not None
    assert detail["turns"][0]["answer"] == "İlk cevap budur."
    assert detail["turns"][0]["pipeline_state"] == state

    assert store.delete_conversation("chat-sidebar") is True
    assert store.get_conversation("chat-sidebar") is None
