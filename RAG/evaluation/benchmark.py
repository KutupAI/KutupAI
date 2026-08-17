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
    id: str
    query: str
    relevant_chunk_ids: Set[str]
    relevant_source_files: Set[str]
    expected_law_numbers: Set[str]
    expected_article_numbers: Set[str]
    notes: str = ""
    split: str = "synthetic"


def load_eval_dataset(path: Path | str | None = None) -> List[EvalExample]:
    path = Path(path or evaluation_config.default_dataset)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw["examples"] if isinstance(raw, dict) and "examples" in raw else raw
    return [
        EvalExample(
            id=str(row.get("id", "")),
            query=str(row["query"]),
            relevant_chunk_ids=set(row.get("relevant_chunk_ids") or []),
            relevant_source_files=set(row.get("relevant_source_files") or []),
            expected_law_numbers={str(value) for value in row.get("expected_law_numbers", [])},
            expected_article_numbers={str(value) for value in row.get("expected_article_numbers", [])},
            notes=str(row.get("notes", "")),
            split=str(row.get("split", "synthetic")),
        )
        for row in items
    ]


def _is_relevant(meta: dict, result_id: str, example: EvalExample) -> bool:
    law = str(meta.get("law_number", ""))
    article = str(meta.get("article_number") or meta.get("article_no", ""))

    # Held-out hukuk soruları tüm dosyaya göre değil, kanun/madde atfına göre
    # değerlendirilir. Bilerek geniş sorularda yalnız kanun etiketi yeterlidir.
    if example.expected_law_numbers:
        if law not in example.expected_law_numbers:
            return False
        if example.expected_article_numbers:
            return article in example.expected_article_numbers
        return True

    # 1. Chunk kimliğiyle doğrudan eşleşme kontrolü.
    if result_id in example.relevant_chunk_ids:
        return True
        
    # 2. Kaynak dosya adıyla eşleşme kontrolü.
    source = str(meta.get("source_file", ""))
    if source in example.relevant_source_files:
        return True
        
    # Geriye uyumluluk için hem article_number hem article_no alanı kontrol edilir.
    for key in example.relevant_chunk_ids:
        if "#" in key:
            sf, art = key.split("#", 1)
            # Eğer kaynak dosya ve madde numarası eşleşiyorsa ilgili sayılır
            if source == sf and (article == art or article in art or art in article):
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
    use_graph: Optional[bool] = None,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    examples = load_eval_dataset(dataset_path)
    if split is not None:
        examples = [example for example in examples if example.split == split]
    if not examples:
        raise ValueError("No evaluation examples matched the requested split.")
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
            use_graph=use_graph,
        )
        ids = [r["id"] for r in results]
        relevant = set(ex.relevant_chunk_ids)
        for r in results:
            if _is_relevant(r["metadata"], r["id"], ex):
                relevant.add(r["id"])
        hit_rank = next((rank for rank, result_id in enumerate(ids, start=1) if result_id in relevant), None)
        ranked_ids.append(ids)
        relevant_sets.append(relevant)
        details.append(
            {
                "id": ex.id,
                "query": ex.query,
                "expected_law_numbers": sorted(ex.expected_law_numbers),
                "expected_article_numbers": sorted(ex.expected_article_numbers),
                "hit_rank": hit_rank,
                "top_source": results[0]["metadata"].get("source_file") if results else None,
                "top_law": results[0]["metadata"].get("law_number") if results else None,
                "top_article": (results[0]["metadata"].get("article_no") or results[0]["metadata"].get("article_number")) if results else None,
                "returned": [
                    {
                        "id": result["id"],
                        "law_number": result["metadata"].get("law_number"),
                        "article_number": result["metadata"].get("article_no") or result["metadata"].get("article_number"),
                        "score": result["score"],
                    }
                    for result in results
                ],
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
    parser.add_argument("--graph-rag", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--output", default=None, help="Write per-question diagnostic JSON.")
    parser.add_argument("--split", default=None, help="Evaluate only a named dataset split, e.g. heldout")
    args = parser.parse_args()
    report = run_benchmark(
        args.dataset,
        mode=args.mode,
        use_prf=False if args.no_prf else None,
        use_reranker=False if args.no_reranker else None,
        expansion_strategy=args.expansion,
        use_graph={"auto": None, "on": True, "off": False}[args.graph_rag],
        split=args.split,
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
