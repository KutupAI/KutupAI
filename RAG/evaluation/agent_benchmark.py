"""Reproducible Qwen Legal-RAG generation, citation, cache, and safety benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from RAG.agent.citations import cited_labels
from RAG.agent.legal_agent import LegalRagAgent


def _vram_mb() -> float | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=5
        )
        return float(output.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _mean(values: List[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="RAG/evaluation/datasets/generation_legal.json")
    parser.add_argument("--output", default="RAG/evaluation/experiments/final/qwen_generation_benchmark.json")
    parser.add_argument("--cache-check", action="store_true")
    parser.add_argument("--limit", type=int, help="Run only the first N cases (diagnostic use).")
    args = parser.parse_args()

    cases = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: max(1, args.limit)]
    agent = LegalRagAgent()
    rows: List[Dict[str, Any]] = []
    vram_before = _vram_mb()
    for case in cases:
        answer = agent.answer(case["query"], use_cache=False)
        sources = answer.sources
        expected = case.get("expected_law_number")
        expected_article = case.get("expected_article_number")
        expected_source = any(
            str(source.get("law_number")) == str(expected)
            and (not expected_article or str(source.get("article_number")) == str(expected_article))
            for source in sources
        )
        labels = cited_labels(answer.answer)
        known = {str(source.get("label")) for source in sources}
        citation_precision = (len([label for label in labels if label in known]) / len(labels)) if labels else 0.0
        rows.append(
            {
                **case,
                "grounded": answer.grounded,
                "refusal_reason": answer.refusal_reason,
                "expected_source_retrieved": expected_source,
                "citation_precision": citation_precision,
                "citation_count": len(labels),
                "retrieval_ms": answer.retrieval_ms,
                "generation_ms": answer.generation_ms,
                "total_ms": answer.total_ms,
                "ttft_ms": answer.ttft_ms,
                "tokens_per_second": answer.tokens_per_second,
            }
        )

    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    cache_result = None
    if args.cache_check and answerable:
        query = answerable[0]["query"]
        agent.answer(query, use_cache=True)  # İlk çağrı cevabı cache'e yazar.
        cache_result = agent.answer(query, use_cache=True).to_dict()
    metrics = {
        "answer_correct_proxy": _mean([float(row["grounded"] and row["expected_source_retrieved"]) for row in answerable]),
        "citation_precision": _mean([float(row["citation_precision"]) for row in answerable]),
        "citation_recall_proxy": _mean([float(row["grounded"] and row["expected_source_retrieved"]) for row in answerable]),
        "hallucination_rate": _mean([float(row["grounded"]) for row in unanswerable]),
        "mean_retrieval_ms": _mean([float(row["retrieval_ms"]) for row in rows]),
        "mean_generation_ms": _mean([float(row["generation_ms"]) for row in rows]),
        "mean_total_ms": _mean([float(row["total_ms"]) for row in rows]),
        "mean_ttft_ms": _mean([float(row["ttft_ms"]) for row in rows if row["ttft_ms"] is not None]),
        "mean_tokens_per_second": _mean([float(row["tokens_per_second"]) for row in rows if row["tokens_per_second"] is not None]),
        "vram_before_mb": vram_before,
        "vram_after_mb": _vram_mb(),
    }
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "rows": rows, "metrics": metrics, "cache_check": cache_result}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
