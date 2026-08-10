from __future__ import annotations

from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class PreprocessingConfig:
    enabled: bool = True
    grayscale: bool = True
    denoise: bool = True
    contrast: bool = True
    sharpen: bool = True
    deskew: bool = True
    perspective_correction: bool = True
    # Far phone photo: crop page + upscale small text
    auto_crop_document: bool = True
    min_dimension: int = 1800
    max_dimension: int = 5000
    # Pale / faded ink & paper
    pale_boost: bool = True
    adaptive_threshold: bool = False
    denoise_kernel: int = 3
    denoise_strength: float = 6.0
    sharpen_amount: float = 1.1
    # Allow stronger phone-tilt correction
    deskew_min_angle: float = 0.4
    deskew_max_angle: float = 35.0


@dataclass(frozen=True)
class OCRConfig:
    language: str = "tr"
    ocr_version: str = "PP-OCRv5"
    device: str = "cpu"
    pipeline_name: str = "PP-StructureV3"
    confidence_threshold: float = 0.35
    low_confidence_threshold: float = 0.55
    max_file_size_mb: int = 100
    max_pdf_pages: int = 500
    # Higher DPI helps distant / low-res phone scans embedded in PDF
    pdf_dpi: int = 300
    enable_layout: bool = True
    enable_tables: bool = True
    enable_visual_elements: bool = True
    enable_doc_orientation: bool = True
    enable_doc_unwarping: bool = True
    enable_textline_orientation: bool = True
    enable_seal_recognition: bool = True
    enable_formula_recognition: bool = False
    # RapidOCR ONNX when Paddle oneDNN fails (common on Windows)
    enable_rapid_fallback: bool = True
    model_dir: str | None = None
    pipeline_config: str | None = None
    cache_dir: str | None = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

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
        return cls(
            language=os.getenv("OCR_LANGUAGE", "tr"),
            ocr_version=os.getenv("OCR_VERSION", "PP-OCRv5"),
            device=os.getenv("OCR_DEVICE", "cpu"),
            pipeline_name=os.getenv("OCR_PIPELINE", "PP-StructureV3"),
            confidence_threshold=float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.35")),
            low_confidence_threshold=float(os.getenv("OCR_LOW_CONFIDENCE_THRESHOLD", "0.55")),
            max_file_size_mb=int(os.getenv("OCR_MAX_FILE_SIZE_MB", "100")),
            max_pdf_pages=int(os.getenv("OCR_MAX_PDF_PAGES", "500")),
            pdf_dpi=int(os.getenv("OCR_PDF_DPI", "300")),
            enable_layout=boolean("OCR_ENABLE_LAYOUT", True),
            enable_tables=boolean("OCR_ENABLE_TABLES", True),
            enable_visual_elements=boolean("OCR_ENABLE_VISUAL_ELEMENTS", True),
            enable_doc_orientation=boolean("OCR_DOC_ORIENTATION", True),
            enable_doc_unwarping=boolean("OCR_DOC_UNWARPING", True),
            enable_textline_orientation=boolean("OCR_TEXTLINE_ORIENTATION", True),
            enable_seal_recognition=boolean("OCR_SEAL_RECOGNITION", True),
            enable_formula_recognition=boolean("OCR_FORMULA_RECOGNITION", False),
            enable_rapid_fallback=boolean("OCR_RAPID_FALLBACK", True),
            model_dir=os.getenv("OCR_MODEL_DIR") or None,
            pipeline_config=os.getenv("OCR_PIPELINE_CONFIG") or None,
            cache_dir=os.getenv("OCR_CACHE_DIR") or None,
            preprocessing=pp,
        )

    def validate(self) -> None:
        if self.language != "tr":
            raise ValueError("OCR Agent is configured for Turkish documents; language must be 'tr'.")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if not 0 <= self.low_confidence_threshold <= 1:
            raise ValueError("low_confidence_threshold must be between 0 and 1.")
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive.")
        if self.max_pdf_pages <= 0:
            raise ValueError("max_pdf_pages must be positive.")
