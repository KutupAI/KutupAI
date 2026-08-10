"""E2E smoke: ingest → hybrid/PRF/rerank → benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Skip CrossEncoder")
    args = parser.parse_args()

    from RAG.client import RetrievalRequest
    from RAG.client.rag_client import get_legal_context
    from RAG.evaluation.benchmark import run_benchmark
    from RAG.ingestion.pipeline import build_vector_database
    from RAG.retriever.bm25_index import get_bm25_index, reset_bm25_singleton
    from RAG.vector_store.chroma_store import reset_vector_store_singleton

    reset_vector_store_singleton()
    reset_bm25_singleton()

    print("1) Ingestion")
    report = build_vector_database(reset=True)
    print(f"   chunks={report.total_chunks} vectors={report.vector_count} bm25={len(get_bm25_index().docs)}")
    if report.total_chunks == 0:
        print("ERROR: no chunks")
        return 1

    print("2) Retrieval")
    resp = get_legal_context(
        RetrievalRequest(query="Is sozlesmesinin feshinde ihbar sureleri nelerdir?", top_k=3, use_reranker=False if args.fast else None)
    )
    top = resp.sources[0] if resp.sources else {}
    print(f"   hits={resp.result_count} top_article={top.get('article_number')} file={top.get('source_file')}")
    if resp.result_count <= 0:
        print("ERROR: no retrieval hits")
        return 1

    print("3) Benchmark")
    metrics = run_benchmark(use_reranker=False if args.fast else None)["metrics"]
    print("  ", json.dumps(metrics, ensure_ascii=False))
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
