from __future__ import annotations

from dataclasses import dataclass, field
import os


def _optional_positive_int(raw: str | None) -> int | None:
    """Parse an optional positive int. Empty / 0 / 'none' / 'unlimited' => None (no cap)."""
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"", "0", "none", "null", "unlimited", "-1"}:
        return None
    parsed = int(value)
    return None if parsed <= 0 else parsed


@dataclass(frozen=True)
class PreprocessingConfig:
    enabled: bool = True
    grayscale: bool = True
    denoise: bool = True
    contrast: bool = True
    sharpen: bool = True
    deskew: bool = True
    perspective_correction: bool = True
    auto_crop_document: bool = True
    min_dimension: int = 1800
    max_dimension: int = 5000
    pale_boost: bool = True
    adaptive_threshold: bool = False
    denoise_kernel: int = 3
    denoise_strength: float = 6.0
    sharpen_amount: float = 1.1
    deskew_min_angle: float = 0.4
    deskew_max_angle: float = 35.0


@dataclass(frozen=True)
class VisionFallbackConfig:
    """PaddleOCR-VL via local llama-server — fallback only, never primary OCR."""

    enabled: bool = True
    provider: str = "paddleocr-vl"
    confidence_threshold: float = 0.45
    max_pages_per_document: int | None = None
    use_shared_inference_client: bool = False
    request_timeout_s: int = 600
    endpoint: str = "http://127.0.0.1:8111/v1/chat/completions"
    model_name: str = "paddleocr-vl-1.6-gguf"
    max_tokens: int = 16384
    band_target_px: int = 1200
    band_overlap: float = 0.12
    max_continue_rounds: int = 12


@dataclass(frozen=True)
class OCRConfig:
    language: str = "tr"
    ocr_version: str = "PP-OCRv5"
    structure_version: str = "PP-StructureV3"
    device: str = "auto"  # auto | cpu | gpu | gpu:N
    performance_profile: str = "development"
    pipeline_name: str = "PP-StructureV3"
    confidence_threshold: float = 0.35
    low_confidence_threshold: float = 0.55
    quality_threshold: float = 0.5
    max_ocr_attempts: int = 3
    max_file_size_mb: int | None = None
    max_pdf_pages: int | None = None
    max_office_pages: int | None = None
    pdf_dpi: int = 300
    enable_layout: bool = True
    enable_tables: bool = True
    enable_visual_elements: bool = True
    enable_doc_orientation: bool = True
    enable_doc_unwarping: bool = True
    enable_textline_orientation: bool = True
    enable_seal_recognition: bool = True
    enable_signature_detection: bool = True
    enable_formula_recognition: bool = False
    enable_rapid_fallback: bool = True
    model_dir: str | None = None
    pipeline_config: str | None = None
    cache_dir: str | None = None
    temp_dir: str | None = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    vision_fallback: VisionFallbackConfig = field(default_factory=VisionFallbackConfig)

    @classmethod
    def from_env(cls) -> "OCRConfig":
        def boolean(name: str, default: bool) -> bool:
            value = os.getenv(name)
            return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}

        pp = PreprocessingConfig(
            enabled=boolean("OCR_PREPROCESSING_ENABLED", True),
            grayscale=boolean("OCR_GRAYSCALE", True),
            denoise=boolean("OCR_DENOISE", True),
            contrast=boolean("OCR_CONTRAST", True),
            sharpen=boolean("OCR_SHARPEN", True),
            deskew=boolean("OCR_DESKEW", True),
            perspective_correction=boolean("OCR_PERSPECTIVE_CORRECTION", True),
            auto_crop_document=boolean("OCR_AUTO_CROP", True),
            pale_boost=boolean("OCR_PALE_BOOST", True),
            adaptive_threshold=boolean("OCR_ADAPTIVE_THRESHOLD", False),
            min_dimension=int(os.getenv("OCR_MIN_DIMENSION", "1800")),
            max_dimension=int(os.getenv("OCR_MAX_DIMENSION", "5000")),
            deskew_max_angle=float(os.getenv("OCR_DESKEW_MAX_ANGLE", "35")),
        )
        vf = VisionFallbackConfig(
            enabled=boolean("OCR_VISION_FALLBACK_ENABLED", True),
            provider=os.getenv("OCR_VISION_FALLBACK_PROVIDER", "paddleocr-vl"),
            confidence_threshold=float(os.getenv("OCR_VISION_FALLBACK_THRESHOLD", "0.45")),
            max_pages_per_document=_optional_positive_int(
                os.getenv("OCR_MAX_VISION_PAGES")
                or os.getenv("OCR_VISION_FALLBACK_MAX_PAGES")
            ),
            use_shared_inference_client=boolean("OCR_VISION_FALLBACK_USE_SHARED_CLIENT", False),
            request_timeout_s=int(os.getenv("OCR_VISION_FALLBACK_TIMEOUT_S", "600")),
            endpoint=(
                os.getenv("OCR_VISION_FALLBACK_ENDPOINT")
                or os.getenv("PADDLEOCR_VL_ENDPOINT")
                or "http://127.0.0.1:8111/v1/chat/completions"
            ),
            model_name=(
                os.getenv("OCR_VISION_FALLBACK_MODEL")
                or os.getenv("PADDLEOCR_VL_MODEL_NAME")
                or "paddleocr-vl-1.6-gguf"
            ),
            max_tokens=int(os.getenv("OCR_VISION_FALLBACK_MAX_TOKENS", "16384")),
            band_target_px=max(400, int(os.getenv("OCR_VISION_FALLBACK_BAND_PX", "1200"))),
            band_overlap=float(os.getenv("OCR_VISION_FALLBACK_BAND_OVERLAP", "0.12")),
            max_continue_rounds=max(1, int(os.getenv("OCR_VISION_FALLBACK_CONTINUE_ROUNDS", "12"))),
        )
        return cls(
            language=os.getenv("OCR_LANGUAGE", "tr"),
            ocr_version=os.getenv("OCR_VERSION", "PP-OCRv5"),
            structure_version=os.getenv("OCR_STRUCTURE_VERSION", "PP-StructureV3"),
            device=os.getenv("OCR_DEVICE", "auto"),
            performance_profile=os.getenv("OCR_PERFORMANCE_PROFILE", "development"),
            pipeline_name=os.getenv("OCR_PIPELINE", "PP-StructureV3"),
            confidence_threshold=float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.35")),
            low_confidence_threshold=float(os.getenv("OCR_LOW_CONFIDENCE_THRESHOLD", "0.55")),
            quality_threshold=float(os.getenv("OCR_QUALITY_THRESHOLD", "0.5")),
            max_ocr_attempts=int(os.getenv("OCR_MAX_ATTEMPTS", "3")),
            max_file_size_mb=_optional_positive_int(os.getenv("OCR_MAX_FILE_SIZE_MB")),
            max_pdf_pages=_optional_positive_int(os.getenv("OCR_MAX_PDF_PAGES")),
            max_office_pages=_optional_positive_int(os.getenv("OCR_MAX_OFFICE_PAGES")),
            pdf_dpi=int(os.getenv("OCR_PDF_DPI", "300")),
            enable_layout=boolean("OCR_ENABLE_LAYOUT", True),
            enable_tables=boolean("OCR_ENABLE_TABLES", True),
            enable_visual_elements=boolean("OCR_ENABLE_VISUAL_ELEMENTS", True),
            enable_doc_orientation=boolean("OCR_DOC_ORIENTATION", True),
            enable_doc_unwarping=boolean("OCR_DOC_UNWARPING", True),
            enable_textline_orientation=boolean("OCR_TEXTLINE_ORIENTATION", True),
            enable_seal_recognition=boolean("OCR_SEAL_RECOGNITION", True),
            enable_signature_detection=boolean("OCR_SIGNATURE_DETECTION", True),
            enable_formula_recognition=boolean("OCR_FORMULA_RECOGNITION", False),
            enable_rapid_fallback=boolean("OCR_RAPID_FALLBACK", True),
            model_dir=os.getenv("OCR_MODEL_DIR") or None,
            pipeline_config=os.getenv("OCR_PIPELINE_CONFIG") or None,
            cache_dir=os.getenv("OCR_CACHE_DIR") or None,
            temp_dir=os.getenv("OCR_TEMP_DIR") or None,
            preprocessing=pp,
            vision_fallback=vf,
        )

    def validate(self) -> None:
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if not 0 <= self.low_confidence_threshold <= 1:
            raise ValueError("low_confidence_threshold must be between 0 and 1.")
        if not 0 <= self.quality_threshold <= 1:
            raise ValueError("quality_threshold must be between 0 and 1.")
        if self.max_ocr_attempts <= 0:
            raise ValueError("max_ocr_attempts must be positive.")
        if self.max_file_size_mb is not None and self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive or None (unlimited).")
        if self.max_pdf_pages is not None and self.max_pdf_pages <= 0:
            raise ValueError("max_pdf_pages must be positive or None (unlimited).")
        if self.vision_fallback.max_pages_per_document is not None and (
            self.vision_fallback.max_pages_per_document <= 0
        ):
            raise ValueError("vision max_pages_per_document must be positive or None.")
        if self.vision_fallback.max_tokens <= 0:
            raise ValueError("vision max_tokens must be positive.")
        if self.performance_profile not in {"development", "production", "high_performance"}:
            raise ValueError(
                "performance_profile must be one of: development, production, high_performance."
            )
