"""
fast_classification_service.py
----------------------------------
The interface classification_agent calls before falling back to Qwen VLM
(Documentation/architecture.md section 7: "reduce load on the AI Inference
Layer via a fast/cheap initial filter/classification step").

No ONNX model is trained/committed yet (Optimization/models/README.md --
the .onnx file is intentionally not in Git). Until one exists, this module
degrades gracefully: classify_fast() returns None, which tools.py in
classification_agent treats as "skip fast path, go straight to Qwen VLM".
This keeps the pipeline runnable end-to-end today, and swapping in a real
ONNX session later only requires filling in `_load_session` /
`_run_session` below -- callers do not change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from Optimization.services.preprocessing import clean_for_fast_classifier

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "classification_model.onnx"


@dataclass(frozen=True)
class FastClassificationResult:
    document_type: str
    confidence: float
    alternatives: list[tuple[str, float]]


def is_model_available() -> bool:
    return _MODEL_PATH.exists()


def classify_fast(text: str) -> FastClassificationResult | None:
    """Run the lightweight ONNX classifier, if a model has been placed at
    Optimization/models/classification_model.onnx. Returns None when no
    model is available (see module docstring) rather than raising, since a
    missing fast-path model is a normal/expected state, not an error --
    classification must still complete via Qwen VLM.
    """
    if not is_model_available():
        return None

    cleaned = clean_for_fast_classifier(text)
    if not cleaned:
        return None

    # NOTE: actual ONNX Runtime session load/run belongs in
    # Optimization/runtime/onnx_runtime_wrapper.py + session_manager.py
    # once a trained classification_model.onnx exists (depends on the
    # labeled dataset from task doc section 6). Wiring is left as a single
    # call here so classification_agent/tools.py never needs to change:
    #
    #   from Optimization.runtime.session_manager import get_session
    #   session = get_session(_MODEL_PATH)
    #   return _run_session(session, cleaned)
    return None
