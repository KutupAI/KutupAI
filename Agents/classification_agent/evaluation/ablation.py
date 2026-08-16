"""
ablation.py
-------------
Ablation/comparison harness required by the task document, section 10.
Qwen's standalone result alone is NOT accepted as sufficient; at minimum
these comparisons must be run on the same test set:

  1. OCR text only
  2. Image/Vision only
  3. OCR text + image
  4. OCR + image + layout info (when available)
  5. Different Qwen model sizes / quantization options
  6. Training with and without class balancing

Every experiment must record Macro-F1, per-class F1, latency, and resource
usage (§10) -- not accuracy alone; the best method is chosen weighing
accuracy AND runtime cost together, not by top score alone.

This module provides the VARIANT definitions and a runner that drives
classification_agent.tools.run_qwen_classification directly (bypassing the
fast-classifier/needs_review logic in agent.py, since ablation cares about
the raw model decision under each input condition, not the production
routing policy). Model-size/quantization and balancing variants are left
as TODO hooks since they depend on artifacts (multiple Qwen ggufs, two
trained model checkpoints) that do not exist yet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from Agents.classification_agent.config import ClassificationConfig
from Agents.classification_agent.dataset.schema import LabeledDocument
from Agents.classification_agent.evaluation.metrics import compute_metrics
from Agents.classification_agent.tools import run_qwen_classification


@dataclass(frozen=True)
class AblationVariant:
    name: str
    tr_description: str
    use_text: bool
    use_image: bool
    use_layout: bool


INPUT_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant("ocr_text_only", "OCR text only", use_text=True, use_image=False, use_layout=False),
    AblationVariant("image_only", "Image/Vision only", use_text=False, use_image=True, use_layout=False),
    AblationVariant("ocr_plus_image", "OCR text + image", use_text=True, use_image=True, use_layout=False),
    AblationVariant(
        "ocr_plus_image_plus_layout",
        "OCR + image + layout bilgisi",
        use_text=True,
        use_image=True,
        use_layout=True,
    ),
)

# TODO once >=2 Qwen ggufs (different size/quantization) are available:
# add one AblationVariant-like config per model + point qwen_model_name /
# qwen_base_url at each, per §10 point 5.
MODEL_SIZE_VARIANTS_TODO: tuple[str, ...] = ()

# TODO once class-balancing (weighting/oversampling, §6) is implemented in
# the training/inference path: run this whole ablation twice, with and
# without balancing enabled, per §10 point 6.
CLASS_BALANCING_TODO = "not yet implemented -- depends on labeled dataset from dataset/ module"


def run_variant(
    variant: AblationVariant,
    record: LabeledDocument,
    *,
    normalized_text: str,
    image_bytes: bytes | None,
    layout: Any,
    ocr_confidence: float | None,
    config: ClassificationConfig,
) -> dict:
    """Run one document through one ablation variant, returning
    {"document_id", "predicted", "confidence", "latency_ms"} or an
    "error" key on failure (kept in the result set, not raised, so one bad
    document doesn't abort the whole ablation run)."""
    text_input = normalized_text if variant.use_text else ""
    image_input = image_bytes if variant.use_image else None
    layout_input = layout if variant.use_layout else None

    variant_config = ClassificationConfig(
        **{**config.__dict__, "send_image": variant.use_image, "use_layout_when_available": variant.use_layout}
    )

    started = time.monotonic()
    try:
        output = run_qwen_classification(
            normalized_text=text_input,
            ocr_confidence=ocr_confidence,
            layout=layout_input,
            image_bytes=image_input,
            config=variant_config,
        )
        elapsed_ms = (time.monotonic() - started) * 1000
        return {
            "document_id": record.document_id,
            "predicted": output["document_type"],
            "confidence": output["confidence"],
            "latency_ms": elapsed_ms,
        }
    except Exception as exc:  # noqa: BLE001 -- ablation must not crash on one bad doc
        return {
            "document_id": record.document_id,
            "predicted": None,
            "confidence": 0.0,
            "latency_ms": (time.monotonic() - started) * 1000,
            "error": str(exc),
        }


def run_ablation(
    records: list[LabeledDocument],
    load_inputs: Callable[[LabeledDocument], dict],
    *,
    config: ClassificationConfig,
    variants: tuple[AblationVariant, ...] = INPUT_VARIANTS,
) -> dict[str, dict]:
    """load_inputs(record) -> {"normalized_text", "image_bytes", "layout",
    "ocr_confidence"} -- left as a caller-supplied function since actually
    reading PDFs/OCR JSON off disk belongs to dataset/loader.py + ocr_agent,
    not to this evaluation module.

    Returns {variant_name: {"metrics": ..., "predictions": [...]}}.
    """
    report: dict[str, dict] = {}

    for variant in variants:
        y_true, y_pred, latencies = [], [], []
        predictions = []
        for record in records:
            if not record.label:
                continue
            inputs = load_inputs(record)
            result = run_variant(
                variant,
                record,
                normalized_text=inputs.get("normalized_text", ""),
                image_bytes=inputs.get("image_bytes"),
                layout=inputs.get("layout"),
                ocr_confidence=inputs.get("ocr_confidence"),
                config=config,
            )
            predictions.append(result)
            y_true.append(record.label)
            y_pred.append(result["predicted"] or "")
            latencies.append(result["latency_ms"])

        report[variant.name] = {
            "description": variant.tr_description,
            "metrics": compute_metrics(y_true, y_pred, latencies),
            "predictions": predictions,
        }

    return report
