"""Reproducible retrieval benchmark: latency, process memory, and exact legal-citation accuracy."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import psutil

from RAG.evaluation.benchmark import _is_relevant, load_eval_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evaluation" / "datasets" / "heldout_legal.json"
DEFAULT_OUTPUT_DIR = ROOT / "evaluation" / "experiments"
DEFAULT_TURBOVEC_DIR = ROOT / "documents" / ".turbovec"
DEFAULT_FAISS_DIR = ROOT / "documents" / ".faiss"


class MemorySampler:
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.peak_rss = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.02):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def __enter__(self) -> "MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bytes_on_disk(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _hardware() -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        data["cuda_available"] = bool(torch.cuda.is_available())
        data["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        data["cuda_available"] = False
        data["gpu"] = None
    return data


def run_system_benchmark(
    *,
    backend: str,
    dataset_path: Path,
    top_k: int,
    warmup: int,
    build_turbovec: bool,
    build_faiss: bool,
    mode: str,
    use_prf: bool,
    use_reranker: bool,
) -> Dict[str, Any]:
    examples = load_eval_dataset(dataset_path)
    examples = [item for item in examples if item.split == "heldout"]
    if not examples:
        raise ValueError("The selected dataset has no heldout examples.")

    index_bytes = 0
    if backend == "chroma":
        from RAG.chroma.chroma_config import chroma_config
        from RAG.retriever.retriever import retrieve

        search: Callable[[str], list] = lambda query: retrieve(
            query, top_k=top_k, mode=mode, use_prf=use_prf, use_reranker=use_reranker
        )
        index_bytes = _bytes_on_disk(Path(chroma_config.persist_directory))
    elif backend == "turbovec":
        from RAG.vector_store.turbovec_store import TurboVecStore
        from RAG.retriever.retriever import retrieve

        store = (
            TurboVecStore.build_from_chroma(DEFAULT_TURBOVEC_DIR)
            if build_turbovec
            else TurboVecStore(DEFAULT_TURBOVEC_DIR)
        )
        search = lambda query: retrieve(query, top_k=top_k, mode=mode, use_prf=use_prf, use_reranker=use_reranker, vector_store=store)
        index_bytes = _bytes_on_disk(DEFAULT_TURBOVEC_DIR)
    elif backend == "faiss":
        from RAG.vector_store.faiss_store import FaissStore
        from RAG.retriever.retriever import retrieve

        store = FaissStore.build_from_chroma(DEFAULT_FAISS_DIR) if build_faiss else FaissStore(DEFAULT_FAISS_DIR)
        search = lambda query: retrieve(query, top_k=top_k, mode=mode, use_prf=use_prf, use_reranker=use_reranker, vector_store=store)
        index_bytes = _bytes_on_disk(DEFAULT_FAISS_DIR)
    else:
        raise ValueError("backend must be chroma, faiss, or turbovec")

    # Embedding modeli, veritabanı bağlantısı ve CPU cache'i ısıtılır; bu süre
    # soru başına gecikme ölçümüne dahil edilmez.
    for example in examples[: min(warmup, len(examples))]:
        search(example.query)

    process = psutil.Process(os.getpid())
    rss_after_load = process.memory_info().rss
    latencies_ms: List[float] = []
    reciprocal_ranks: List[float] = []
    hit_counts = {1: 0, 3: 0, 5: 0}
    failures: List[Dict[str, Any]] = []

    started = time.perf_counter()
    with MemorySampler() as memory:
        for example in examples:
            started_query = time.perf_counter()
            results = search(example.query)
            latencies_ms.append((time.perf_counter() - started_query) * 1000)
            rank = 0
            for position, result in enumerate(results, start=1):
                if _is_relevant(result["metadata"], result["id"], example):
                    rank = position
                    break
            if rank:
                reciprocal_ranks.append(1.0 / rank)
                for cutoff in hit_counts:
                    if rank <= cutoff:
                        hit_counts[cutoff] += 1
            else:
                reciprocal_ranks.append(0.0)
                failures.append(
                    {
                        "id": example.id,
                        "query": example.query,
                        "expected_law_numbers": sorted(example.expected_law_numbers),
                        "expected_article_numbers": sorted(example.expected_article_numbers),
                        "returned": [
                            {
                                "law_number": item["metadata"].get("law_number"),
                                "article_number": item["metadata"].get("article_number") or item["metadata"].get("article_no"),
                                "source_file": item["metadata"].get("source_file"),
                            }
                            for item in results
                        ],
                    }
                )
    elapsed = time.perf_counter() - started
    n = len(examples)
    return {
        "benchmark": "turkish_legal_heldout_retrieval_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "dataset": str(dataset_path),
        "n_queries": n,
        "top_k": top_k,
        "retrieval_configuration": {"mode": mode, "prf": use_prf, "reranker": use_reranker},
        "hardware": _hardware(),
        "accuracy": {
            "MRR": round(statistics.mean(reciprocal_ranks), 4),
            **{f"Hit@{cutoff}": round(count / n, 4) for cutoff, count in hit_counts.items()},
        },
        "search_speed": {
            "total_seconds": round(elapsed, 4),
            "queries_per_second": round(n / elapsed, 3),
            "mean_ms": round(statistics.mean(latencies_ms), 3),
            "p50_ms": round(_percentile(latencies_ms, 50), 3),
            "p95_ms": round(_percentile(latencies_ms, 95), 3),
            "p99_ms": round(_percentile(latencies_ms, 99), 3),
        },
        "memory": {
            "rss_after_load_mb": round(rss_after_load / 1024 / 1024, 2),
            "peak_rss_mb": round(memory.peak_rss / 1024 / 1024, 2),
            "index_on_disk_mb": round(index_bytes / 1024 / 1024, 2),
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["chroma", "faiss", "turbovec"], default="chroma")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mode", choices=["vector", "hybrid", "bm25"], default="vector")
    parser.add_argument("--prf", action="store_true", help="Enable pseudo-relevance feedback (Chroma only).")
    parser.add_argument("--reranker", action="store_true", help="Enable the configured reranker (Chroma only).")
    parser.add_argument("--build-turbovec", action="store_true")
    parser.add_argument("--build-faiss", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_system_benchmark(
        backend=args.backend,
        dataset_path=args.dataset,
        top_k=args.top_k,
        warmup=args.warmup,
        build_turbovec=args.build_turbovec,
        build_faiss=args.build_faiss,
        mode=args.mode,
        use_prf=args.prf,
        use_reranker=args.reranker,
    )
    output = args.output or DEFAULT_OUTPUT_DIR / f"{args.backend}_heldout_baseline.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("backend", "n_queries", "accuracy", "search_speed", "memory")}, ensure_ascii=False, indent=2))
    print(f"Saved benchmark: {output}")


if __name__ == "__main__":
    main()
