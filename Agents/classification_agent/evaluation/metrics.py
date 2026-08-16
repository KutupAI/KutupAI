"""
metrics.py
------------
Evaluation metrics required by the task document, section 8:
Accuracy, Macro-F1 (primary metric -- imbalance-robust), Weighted-F1,
per-class Precision/Recall/F1, and a full confusion matrix.

Pure-python, no sklearn dependency (mirrors RAG/evaluation/metrics.py's
style in this project). Always reports EVERY taxonomy class, even with
zero support, so a missing/never-predicted class is visible in the report
rather than silently absent.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

from Agents.classification_agent.taxonomy import DOCUMENT_CLASSES

ALL_CODES: tuple[str, ...] = tuple(c.code for c in DOCUMENT_CLASSES)


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = ALL_CODES) -> dict[str, dict[str, int]]:
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return matrix


def per_class_prf(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = ALL_CODES) -> dict[str, dict[str, float]]:
    support = Counter(y_true)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1

    result = {}
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        result[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support.get(label, 0),
        }
    return result


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return round(correct / len(y_true), 4)


def macro_f1(per_class: dict[str, dict[str, float]], *, only_supported: bool = True) -> float:
    """§8 primary metric. `only_supported=True` (default) excludes classes
    with zero real examples from the average, since an F1 of 0 for a class
    that literally has no test examples is not a model failure -- it is a
    data-coverage gap. Set False to see the raw (harsher) number too.
    """
    scores = [c["f1"] for c in per_class.values() if (not only_supported) or c["support"] > 0]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def weighted_f1(per_class: dict[str, dict[str, float]]) -> float:
    total_support = sum(c["support"] for c in per_class.values())
    if not total_support:
        return 0.0
    weighted = sum(c["f1"] * c["support"] for c in per_class.values())
    return round(weighted / total_support, 4)


def latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    """§8 requires latency alongside accuracy metrics."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "n": 0}
    ordered = sorted(latencies_ms)
    n = len(ordered)

    def percentile(p: float) -> float:
        idx = min(n - 1, int(round(p * (n - 1))))
        return ordered[idx]

    return {
        "mean_ms": round(sum(ordered) / n, 2),
        "p50_ms": round(percentile(0.50), 2),
        "p95_ms": round(percentile(0.95), 2),
        "max_ms": round(ordered[-1], 2),
        "n": n,
    }


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str], latencies_ms: Sequence[float] = ()) -> dict:
    """One-call entry point producing everything §8 asks for."""
    per_class = per_class_prf(y_true, y_pred)
    return {
        "n_examples": len(y_true),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(per_class),
        "macro_f1_all_classes": macro_f1(per_class, only_supported=False),
        "weighted_f1": weighted_f1(per_class),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "latency": latency_stats(latencies_ms),
    }
