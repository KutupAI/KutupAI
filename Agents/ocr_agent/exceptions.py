class OCRAgentError(Exception):
    """Base exception for all OCR Agent failures."""


class UnsupportedDocumentError(OCRAgentError):
    pass


class DocumentReadError(OCRAgentError):
    pass


class PreprocessingError(OCRAgentError):
    pass


class OCREngineError(OCRAgentError):
    pass


class OCRValidationError(OCRAgentError):
    pass
