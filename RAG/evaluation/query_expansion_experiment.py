"""Compare query-expansion strategies and keep the best in rag_config.yaml."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from RAG.configuration.rag_config_loader import query_expansion_config
from RAG.evaluation.benchmark import run_benchmark

_CONFIG_YAML = Path(__file__).resolve().parents[1] / "configuration" / "rag_config.yaml"
_RESULTS_PATH = Path(__file__).resolve().parent / "datasets" / "qe_experiment_results.json"


def _score(metrics: Dict[str, float]) -> float:
    return 3 * float(metrics.get("Hit@1", 0)) + 2 * float(metrics.get("MRR", 0)) + float(metrics.get("Hit@3", 0))


def run_query_expansion_experiment(
    *,
    dataset_path: Path | str | None = None,
    strategies: Optional[List[str]] = None,
    update_config: bool = True,
) -> Dict[str, Any]:
    rows = []
    for strategy in strategies or list(query_expansion_config.strategies):
        metrics = run_benchmark(dataset_path, expansion_strategy=strategy)["metrics"]
        rows.append({"strategy": strategy, "metrics": metrics, "score": round(_score(metrics), 4)})

    rows.sort(key=lambda r: r["score"], reverse=True)
    winner = rows[0]["strategy"] if rows else "none"

    if update_config and rows:
        with open(_CONFIG_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data.setdefault("query_expansion", {})
        data["query_expansion"]["selected_strategy"] = winner
        data["query_expansion"]["enabled"] = winner not in ("none", "off", "")
        with open(_CONFIG_YAML, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    payload = {"winner": winner, "results": rows}
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--no-update-config", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        run_query_expansion_experiment(
            dataset_path=args.dataset,
            update_config=not args.no_update_config,
        ),
        ensure_ascii=False,
        indent=2,
    ))
