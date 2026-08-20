"""
run_test_set.py
------------------
Runs the real ClassificationAgent (Qwen VLM) over every document in a test
manifest (e.g. dataset/splits/test.csv), collects predictions, and computes
the §8 metrics. This is the actual evaluation step -- everything before it
(agent.py, dataset/, metrics.py) exists so this script has something to run.

No OCR text is used here on purpose: reusing ocr_agent for 39 documents
would add a slow, heavy dependency (PaddleOCR) for a one-off evaluation
run. classification_agent already supports image-only input (§4's
"image-only" ablation variant is exactly this path), so this script sends
each document's rendered image straight to Qwen. PDF pages are rendered
with the same PyMuPDF-based renderer ocr_agent already uses
(Agents/ocr_agent/processing/pdf_renderer.py) -- no new rendering code.

Usage:
    python -m Agents.classification_agent.evaluation.run_test_set \
        --test-manifest Agents/classification_agent/dataset/splits/test.csv \
        --output Agents/classification_agent/evaluation/test_results.json
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

from PIL import Image

MAX_IMAGE_DIMENSION = 1600


def load_as_png_bytes(path: str, max_dim: int = MAX_IMAGE_DIMENSION) -> bytes:
    """Load a PDF (first page) or image file and return PNG bytes, resized
    to fit within max_dim on the longer side -- keeps requests fast and
    matches ClassificationConfig.max_image_dimension."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        from Agents.ocr_agent.processing.pdf_renderer import PDFRenderer

        renderer = PDFRenderer(dpi=150, max_pages=500)
        pages = renderer.render(p)
        if not pages:
            raise ValueError(f"PDF has no pages: {path}")
        arr = pages[0][:, :, ::-1]  # BGR -> RGB
        img = Image.fromarray(arr)
    else:
        img = Image.open(p)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

    img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_evaluation(test_manifest_csv: str, output_json: str) -> dict:
    from Agents.classification_agent.agent import ClassificationAgent
    from Agents.classification_agent.config import ClassificationConfig
    from Agents.classification_agent.dataset.loader import load_manifest
    from Agents.classification_agent.evaluation.metrics import compute_metrics

    records = load_manifest(test_manifest_csv, require_labels=True)
    agent = ClassificationAgent(config=ClassificationConfig.from_env())

    y_true: list[str] = []
    y_pred: list[str] = []
    latencies: list[float] = []
    per_doc: list[dict] = []
    skipped: list[dict] = []

    total = len(records)
    for i, record in enumerate(records, start=1):
        print(f"[{i}/{total}] {record.document_id} ({record.label}) ...", flush=True)
        try:
            image_bytes = load_as_png_bytes(record.pdf_path)
        except Exception as exc:
            print(f"    SKIPPED (could not read file): {exc}")
            skipped.append({"document_id": record.document_id, "path": record.pdf_path, "error": str(exc)})
            continue

        state = {"document_id": record.document_id, "document_image": image_bytes}
        result = agent.run(state)["classification_result"]

        predicted = result.get("document_type") or ""
        confidence = result.get("confidence")
        status = result.get("status")
        elapsed_s = (result.get("processing_ms") or 0) / 1000

        correct = "OK" if predicted == record.label else "WRONG"
        print(f"    true={record.label} pred={predicted} conf={confidence} status={status} ({elapsed_s:.1f}s) [{correct}]")

        y_true.append(record.label)
        y_pred.append(predicted)
        latencies.append(result.get("processing_ms") or 0)
        per_doc.append(
            {
                "document_id": record.document_id,
                "true": record.label,
                "predicted": predicted,
                "confidence": confidence,
                "status": status,
                "is_synthetic": record.is_synthetic,
                "processing_ms": result.get("processing_ms"),
            }
        )

        # Save incrementally after every document so a crash/interrupt on
        # doc #30 doesn't lose the first 29 -- important for a run this long.
        _save(output_json, y_true, y_pred, latencies, per_doc, skipped, done=False)

    metrics = compute_metrics(y_true, y_pred, latencies)
    _save(output_json, y_true, y_pred, latencies, per_doc, skipped, done=True, metrics=metrics)

    print("\n=== SONUÇLAR ===")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Macro-F1: {metrics['macro_f1']}")
    print(f"Weighted-F1: {metrics['weighted_f1']}")
    print(f"Latency (mean/p50/p95): {metrics['latency']['mean_ms']}ms / {metrics['latency']['p50_ms']}ms / {metrics['latency']['p95_ms']}ms")
    if skipped:
        print(f"Skipped {len(skipped)} document(s) -- see '{output_json}' -> 'skipped'.")

    return metrics


def _save(output_json, y_true, y_pred, latencies, per_doc, skipped, *, done: bool, metrics: dict | None = None) -> None:
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "done": done,
        "n_processed": len(per_doc),
        "n_skipped": len(skipped),
        "per_document": per_doc,
        "skipped": skipped,
        "metrics": metrics,
    }
    Path(output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run ClassificationAgent over a test manifest and compute metrics.")
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output", default="Agents/classification_agent/evaluation/test_results.json")
    args = parser.parse_args()
    run_evaluation(args.test_manifest, args.output)


if __name__ == "__main__":
    _cli()