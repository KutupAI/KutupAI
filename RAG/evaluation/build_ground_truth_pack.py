"""Create a reviewer-ready legal Q&A pack from human-written eval questions.

The answer is *evidence*, not an unverified LLM summary: every question is
paired with the indexed passages and citation metadata of its expected law and
article. A legal reviewer can approve a short answer against this evidence.

Usage:
    python -m RAG.evaluation.build_ground_truth_pack
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from RAG.evaluation.benchmark import load_eval_dataset
from RAG.vector_store.chroma_store import get_vector_store


def build_pack(dataset: str | None, output: str, *, max_evidence_chunks: int = 2) -> Path:
    by_citation: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in get_vector_store().export_all():
        meta = dict(row.get("metadata") or {})
        key = (str(meta.get("law_number") or ""), str(meta.get("article_no") or meta.get("article_number") or ""))
        if all(key):
            by_citation[key].append(row)

    records = []
    missing = []
    for example in load_eval_dataset(dataset):
        evidence = []
        for law in example.expected_law_numbers:
            articles = example.expected_article_numbers or {""}
            for article in articles:
                candidates = by_citation.get((str(law), str(article)), [])
                for row in candidates[:max_evidence_chunks]:
                    meta = dict(row.get("metadata") or {})
                    evidence.append({
                        "chunk_id": str(row.get("chunk_id") or meta.get("chunk_id") or ""),
                        "law_number": str(meta.get("law_number") or law),
                        "article_number": str(meta.get("article_no") or meta.get("article_number") or article),
                        "source_file": str(meta.get("source_file") or ""),
                        "page_start": meta.get("page_start", meta.get("page")),
                        "page_end": meta.get("page_end", meta.get("page_start", meta.get("page"))),
                        "evidence_text": " ".join(str(row.get("text") or "").split()),
                    })
        if not evidence:
            missing.append(example.id)
        records.append({
            "id": example.id,
            "question": example.query,
            "expected_law_numbers": sorted(example.expected_law_numbers),
            "expected_article_numbers": sorted(example.expected_article_numbers),
            "difficulty": getattr(example, "difficulty", None),
            "question_type": getattr(example, "type", None),
            "gold_answer_evidence": evidence,
            "review_status": "needs_legal_review",
            "reviewer_task": (
                "Evidence passages based on an indexed official text. Write or approve a concise "
                "answer only if it is supported by these passages; retain the cited law/article/page."
            ),
        })

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "dataset": str(dataset or "default"),
        "records": records,
        "records_without_evidence": missing,
        "warning": "Evidence is corpus-grounded; a legal reviewer must approve any natural-language answer.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="RAG/evaluation/datasets/heldout_legal.json")
    parser.add_argument("--output", default="RAG/evaluation/datasets/heldout_legal_ground_truth.json")
    parser.add_argument("--max-evidence-chunks", type=int, default=2)
    args = parser.parse_args()
    print(build_pack(args.dataset, args.output, max_evidence_chunks=max(1, args.max_evidence_chunks)))


if __name__ == "__main__":
    main()
