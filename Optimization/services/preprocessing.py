# Preprocessing helpers for the fast classification service.
#
# clean_for_fast_classifier() was referenced by fast_classification_service.py
# but never implemented, which broke the import chain for the ENTIRE
# classification_agent (any missing function in an imported module fails
# the whole `from ... import` statement, not just calls to that function) --
# even though classify_fast() itself always returns None today anyway (no
# ONNX model exists yet, see that module's docstring). This is a minimal,
# safe placeholder: whitespace normalization only, no behavior change once
# a real ONNX preprocessing pipeline is implemented here later (that
# implementation depends on how the eventual classification_model.onnx
# expects its input text formatted -- lowercasing, truncation length,
# Turkish-specific normalization, etc. -- none of which exists yet since no
# model is trained/committed).


def clean_for_fast_classifier(text: str) -> str:
    """Minimal safe normalization for the (not-yet-trained) fast ONNX
    classifier's input text. Currently just whitespace-collapses and
    strips -- replace with the real preprocessing this model expects once
    classification_model.onnx exists (see fast_classification_service.py).
    """
    if not text:
        return ""
    return " ".join(text.split())