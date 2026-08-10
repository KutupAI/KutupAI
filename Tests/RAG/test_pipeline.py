"""Lightweight RAG unit checks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_load_and_split():
    from RAG.ingestion.chunker import split_documents
    from RAG.ingestion.enricher import enrich_documents
    from RAG.ingestion.loader import load_all_sources
    from RAG.metadata.registry import apply_source_metadata

    docs = load_all_sources()
    assert docs
    chunks = enrich_documents(split_documents(apply_source_metadata(docs)))
    assert chunks and all("chunk_id" in c.metadata for c in chunks)
    print(f"OK load_and_split docs={len(docs)} chunks={len(chunks)}")


def test_metrics():
    from RAG.evaluation.metrics import aggregate_metrics

    m = aggregate_metrics([["a", "b", "c"], ["x", "y", "z"]], [{"b"}, {"z"}], (1, 2, 3))
    assert m["Hit@2"] == 0.5 and "MRR" in m
    print(f"OK metrics {m}")


def test_client_contract():
    from RAG.client import RetrievalRequest, RetrievalResponse

    assert RetrievalRequest(query="q", mode="hybrid").mode == "hybrid"
    assert RetrievalResponse(context="x", result_count=0).result_count == 0
    print("OK client_contract")


if __name__ == "__main__":
    test_load_and_split()
    test_metrics()
    test_client_contract()
    print("All lightweight RAG tests passed.")
