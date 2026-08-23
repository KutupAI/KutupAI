class ClassificationAgentError(Exception):
    """Base exception for all Classification Agent failures."""


class MissingInputError(ClassificationAgentError):
    """Neither OCR text nor a document image was available in state."""


class FastClassifierError(ClassificationAgentError):
    """Optimization layer (ONNX fast classifier) call failed."""


class VLMError(ClassificationAgentError):
    """Classification VLM (Gemma 3, local llama.cpp/llama-server) inference
    call failed. Renamed from QwenVLMError -- see vlm_client.py's module
    docstring for the Qwen -> Gemma 3 migration note. QwenVLMError is kept
    below as a backward-compat alias."""


# Backward-compat alias for any not-yet-updated caller.
QwenVLMError = VLMError


class InvalidClassificationOutputError(ClassificationAgentError):
    """Model output was not valid/parsable JSON, or used a class outside taxonomy.py."""