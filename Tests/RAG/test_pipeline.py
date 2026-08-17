"""Corpus değiştirmeden RAG sözleşmelerini doğrulayan hızlı regresyon testleri."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_load_and_split():
    """Kaynaklar, metadata ve kararlı chunk kimlikleri birlikte üretilir.

    Tam PDF corpusunu okumak bu unit testin görevi değildir; o çalışma
    ``run_retrieval_evaluation.py`` ile ölçülür. Küçük fixture test suite'i
    saniyeler içinde tutar ve CI'da ağ/model bağımlılığını ortadan kaldırır.
    """
    from RAG.ingestion.chunker import split_documents
    from RAG.ingestion.enricher import enrich_documents
    from RAG.metadata.registry import apply_source_metadata

    docs = [
        Document(
            page_content=(
                "KANUN NUMARASI: 9000\n"
                "MADDE 1 – Örnek hukuk hükmü ve uygulama amacı.\n"
                "(1) Birinci fıkra açıklaması.\n\n"
                "MADDE 2 – İkinci örnek hüküm.\n"
                "(1) İkinci fıkra açıklaması."
            ),
            metadata={"source_file": "9000_Ornek_Kanun.pdf", "page_start": 1, "page_end": 1},
        )
    ]
    chunks = enrich_documents(split_documents(apply_source_metadata(docs)))
    assert chunks and all("chunk_id" in c.metadata for c in chunks)
    print(f"OK load_and_split docs={len(docs)} chunks={len(chunks)}")


def test_metrics():
    """Hit@k ve MRR hesaplaması bilinen küçük örnekte korunur."""
    from RAG.evaluation.metrics import aggregate_metrics

    m = aggregate_metrics([["a", "b", "c"], ["x", "y", "z"]], [{"b"}, {"z"}], (1, 2, 3))
    assert m["Hit@2"] == 0.5 and "MRR" in m
    print(f"OK metrics {m}")


def test_client_contract():
    """RAG istemci veri sözleşmesi uygulama katmanıyla uyumludur."""
    from RAG.client import RetrievalRequest, RetrievalResponse

    assert RetrievalRequest(query="q", mode="hybrid").mode == "hybrid"
    assert RetrievalResponse(context="x", result_count=0).result_count == 0
    print("OK client_contract")


if __name__ == "__main__":
    test_load_and_split()
    test_metrics()
    test_client_contract()
    print("All lightweight RAG tests passed.")
