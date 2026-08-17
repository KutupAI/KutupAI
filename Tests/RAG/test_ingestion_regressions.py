"""Model indirmeden çalışan ingestion bütünlük regresyon testleri."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RAG.ingestion.loader import _merge_pdf_pages
from RAG.ingestion.chunker import _resolve_law_number
from RAG.ingestion.pipeline import _unique_chunks
from RAG.retriever.text_utils import tokenize


class IngestionRegressionTests(unittest.TestCase):
    def test_numbered_law_filename_overrides_bad_metadata(self) -> None:
        metadata = {
            "source_file": "1076_Yedek Subaylar Kanunu.pdf",
            "law_number": "222222222",
        }

        self.assertEqual(_resolve_law_number(metadata), "1076")

    def test_law_number_is_read_from_a_document_header_when_filename_has_none(self) -> None:
        metadata = {"source_file": "mevzuat.pdf", "law_number": "unknown"}

        self.assertEqual(
            _resolve_law_number(metadata, "KANUN NUMARASI: 6698\nKişisel Verilerin Korunması Kanunu"),
            "6698",
        )

    def test_ascii_and_turkish_queries_share_bm25_tokens(self) -> None:
        self.assertEqual(
            tokenize("İş sözleşmesinin feshi ve ihbar süreleri"),
            tokenize("Is sozlesmesinin feshi ve ihbar sureleri"),
        )

    def test_pdf_pages_are_merged_in_page_order(self) -> None:
        pages = [
            Document(page_content="Madde 2- Başlangıç", metadata={"source": "law.pdf", "page": 1}),
            Document(page_content="Devam eden hüküm.", metadata={"source": "law.pdf", "page": 2}),
            Document(page_content="Madde 1- Ayrı belge", metadata={"source": "other.pdf", "page": 0}),
        ]

        merged = _merge_pdf_pages(pages)

        self.assertEqual(len(merged), 2)
        law = next(doc for doc in merged if doc.metadata["source"] == "law.pdf")
        # Sayfa işaretleri, daha sonra chunk'ın gerçek PDF sayfasına geri
        # bağlanmasını sağlar; metin birleştirilirken korunmaları zorunludur.
        self.assertEqual(
            law.page_content,
            "[[RAG_PAGE:2]]\nMadde 2- Başlangıç\n\n[[RAG_PAGE:3]]\nDevam eden hüküm.",
        )
        self.assertEqual((law.metadata["page_start"], law.metadata["page_end"]), (2, 3))

    def test_deduplication_keeps_the_first_stable_chunk_id(self) -> None:
        chunks = [
            Document(page_content="first", metadata={"chunk_id": "same"}),
            Document(page_content="duplicate", metadata={"chunk_id": "same"}),
            Document(page_content="other", metadata={"chunk_id": "other"}),
        ]

        unique = _unique_chunks(chunks)

        self.assertEqual([chunk.metadata["chunk_id"] for chunk in unique], ["same", "other"])
        self.assertEqual(unique[0].page_content, "first")


if __name__ == "__main__":
    unittest.main()
