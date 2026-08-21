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
from RAG.ingestion.chunker import split_articles, split_documents, _resolve_law_number
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

    def test_structurally_rejoined_chunk_keeps_its_real_pdf_page(self) -> None:
        """Satır sonu değişse bile sayfa, belge başındaki 1'e düşmemelidir."""
        clause_a = "a) " + ("Birinci bent hükmü. " * 18)
        clause_b = "b) " + ("İkinci bent hükmü. " * 18)
        clause_c = "c) " + ("Üçüncü bent hükmü. " * 18)
        document = Document(
            page_content=(
                "[[RAG_PAGE:1]]\nMadde 1- İlk sayfadaki kısa hüküm.\n\n"
                "[[RAG_PAGE:2]]\nMadde 2- İkinci sayfadaki uzun hüküm.\n"
                f"{clause_a}\n{clause_b}\n{clause_c}"
            ),
            metadata={"source_file": "9000_Ornek_Kanun.pdf", "page_start": 1},
        )

        chunks = split_documents([document])
        second_article_chunks = [
            chunk for chunk in chunks if chunk.metadata.get("article_no") == "2"
        ]

        self.assertTrue(second_article_chunks)
        self.assertTrue(all(chunk.metadata["page_start"] == 2 for chunk in second_article_chunks))

    def test_section_heading_and_additional_article_keep_correct_owner(self) -> None:
        """Başlık sonraki maddeye, EK MADDE ise normal maddeye değil kendine bağlanır."""
        document = Document(
            page_content=(
                "MADDE 121 – Önceki maddenin hükmü.\n\n"
                "Atıf yapılan hükümler\n"
                "MADDE 122 – Eski kanuna yapılan atıflar bu Kanuna yapılmış sayılır.\n\n"
                "Göçmen kaçakçılığı suçunda kullanılan araca elkoyma\n"
                "EK MADDE 1 – Araçlara ilgili hükme göre elkonulur."
            ),
            metadata={"source_file": "6458_Ornek.pdf"},
        )

        articles = split_articles(document)

        self.assertEqual([article["article_no"] for article in articles], ["121", "122", "Ek Madde 1"])
        self.assertNotIn("Atıf yapılan hükümler", articles[0]["content"])
        self.assertTrue(articles[1]["content"].startswith("Atıf yapılan hükümler"))
        self.assertTrue(articles[2]["content"].startswith("Göçmen kaçakçılığı"))

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
