class ClassificationAgentError(Exception):
    """Base exception for all Classification Agent failures."""


class MissingInputError(ClassificationAgentError):
    """Neither OCR text nor a document image was available in state."""


class FastClassifierError(ClassificationAgentError):
    """Optimization layer (ONNX fast classifier) call failed."""


class QwenVLMError(ClassificationAgentError):
    """Qwen VLM inference call failed."""


class InvalidClassificationOutputError(ClassificationAgentError):
    """Model output was not valid/parsable JSON, or used a class outside taxonomy.py."""
