class OCRAgentError(Exception):
    """Base exception for all OCR Agent failures.

    `code` matches the stable error-code vocabulary consumed by
    Orchestration / logging (see README "Error codes").
    """

    code: str = "OCR_AGENT_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


class UnsupportedDocumentError(OCRAgentError):
    code = "UNSUPPORTED_FILE_TYPE"


class DocumentReadError(OCRAgentError):
    code = "FILE_CORRUPTED"


class PreprocessingError(OCRAgentError):
    code = "PAGE_EXTRACTION_FAILED"


class OCREngineError(OCRAgentError):
    code = "OCR_FAILED"


class OCRValidationError(OCRAgentError):
    code = "FILE_CORRUPTED"


class LowImageQualityError(OCRAgentError):
    code = "LOW_IMAGE_QUALITY"


class LowConfidenceError(OCRAgentError):
    code = "LOW_OCR_CONFIDENCE"


class VisionFallbackError(OCRAgentError):
    code = "VISION_FALLBACK_FAILED"


class SignatureDetectionError(OCRAgentError):
    code = "SIGNATURE_DETECTION_FAILED"


class SealDetectionError(OCRAgentError):
    code = "SEAL_DETECTION_FAILED"


class PageExtractionError(OCRAgentError):
    code = "PAGE_EXTRACTION_FAILED"
