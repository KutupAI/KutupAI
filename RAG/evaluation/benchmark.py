"""Retrieval benchmark (Hit@1/2/3 + MRR)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from RAG.configuration.rag_config_loader import evaluation_config, retrieval_config
from RAG.evaluation.metrics import aggregate_metrics
from RAG.retriever.retriever import retrieve


@dataclass
class EvalExample:
    query: str
    relevant_chunk_ids: Set[str]
    relevant_source_files: Set[str]
    notes: str = ""


def load_eval_dataset(path: Path | str | None = None) -> List[EvalExample]:
    path = Path(path or evaluation_config.default_dataset)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw["examples"] if isinstance(raw, dict) and "examples" in raw else raw
    return [
        EvalExample(
            query=str(row["query"]),
            relevant_chunk_ids=set(row.get("relevant_chunk_ids") or []),
            relevant_source_files=set(row.get("relevant_source_files") or []),
            notes=str(row.get("notes", "")),
        )
        for row in items
    ]


def _is_relevant(meta: dict, result_id: str, example: EvalExample) -> bool:
    if result_id in example.relevant_chunk_ids:
        return True
    source = str(meta.get("source_file", ""))
    if source in example.relevant_source_files:
        return True
    article = str(meta.get("article_number", ""))
    for key in example.relevant_chunk_ids:
        if "#" in key:
            sf, art = key.split("#", 1)
            if source == sf and article == art:
                return True
    return False


def run_benchmark(
    dataset_path: Path | str | None = None,
    *,
    top_k: Optional[int] = None,
    mode: Optional[str] = None,
    use_prf: Optional[bool] = None,
    use_reranker: Optional[bool] = None,
    expansion_strategy: Optional[str] = None,
) -> Dict[str, Any]:
    examples = load_eval_dataset(dataset_path)
    k = min(top_k or max(evaluation_config.hit_ks), retrieval_config.max_top_k)

    ranked_ids: List[List[str]] = []
    relevant_sets: List[Set[str]] = []
    details: List[Dict[str, Any]] = []

    for ex in examples:
        results = retrieve(
            ex.query,
            top_k=k,
            mode=mode,
            use_prf=use_prf,
            use_reranker=use_reranker,
            expansion_strategy=expansion_strategy,
        )
        ids = [r["id"] for r in results]
        relevant = set(ex.relevant_chunk_ids)
        for r in results:
            if _is_relevant(r["metadata"], r["id"], ex):
                relevant.add(r["id"])
        ranked_ids.append(ids)
        relevant_sets.append(relevant)
        details.append(
            {
                "query": ex.query,
                "top_source": results[0]["metadata"].get("source_file") if results else None,
                "top_article": results[0]["metadata"].get("article_number") if results else None,
            }
        )

    return {
        "metrics": aggregate_metrics(ranked_ids, relevant_sets, hit_ks=evaluation_config.hit_ks),
        "details": details,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--no-prf", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--expansion", default=None)
    args = parser.parse_args()
    report = run_benchmark(
        args.dataset,
        mode=args.mode,
        use_prf=False if args.no_prf else None,
        use_reranker=False if args.no_reranker else None,
        expansion_strategy=args.expansion,
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
