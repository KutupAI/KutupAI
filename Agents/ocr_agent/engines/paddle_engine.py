"""OCR engines: PaddleOCR → RapidOCR → (VL in processor) → PP-StructureV3 tables.

Heavy PP-StructureV3 companion models are not loaded for normal text OCR:
  PP-OCRv5_server_det, PP-OCRv5_server_rec, latin_PP-OCRv5_mobile_rec,
  PP-LCNet_x1_0_doc_ori, PP-DocLayout_plus-L.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Literal

import numpy as np

from Agents.ocr_agent.config import OCRConfig
from Agents.ocr_agent.device import resolve_device
from Agents.ocr_agent.exceptions import OCREngineError

# Disable oneDNN/MKLDNN early (PaddleX may enable them on CPU otherwise).
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

logger = logging.getLogger(__name__)

EngineKind = Literal["paddle", "rapid", "structure_table"]

# Models explicitly stopped for this layer (not loaded / not forced).
_STOPPED_STRUCTURE_MODELS = frozenset({
    "PP-OCRv5_server_det",
    "PP-OCRv5_server_rec",
    "latin_PP-OCRv5_mobile_rec",
    "PP-LCNet_x1_0_doc_ori",
    "PP-DocLayout_plus-L",
})


class PaddleStructureEngine:
    """Lazy multi-engine OCR: PaddleOCR (primary), RapidOCR, StructureV3 tables."""

    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._paddle: Any = None
        self._rapid: Any = None
        self._structure: Any = None
        self._structure_backend: str | None = None  # table_v2 | structure_v3
        self._mode: EngineKind = "paddle"
        self.engine_name = "PaddleOCR"
        self.last_engine_name = self.engine_name
        self._resolved_device = resolve_device(config.device)
        logger.info(
            "OCR engines ready (lang=%s, device=%s, profile=%s); "
            "stopped Structure models: %s",
            _paddle_language(config.language),
            self._resolved_device,
            config.performance_profile,
            ", ".join(sorted(_STOPPED_STRUCTURE_MODELS)),
        )

    # ------------------------------------------------------------------
    # Public predict API
    # ------------------------------------------------------------------
    def predict(
        self,
        image: np.ndarray,
        *,
        engine: EngineKind | None = None,
    ) -> list[Any]:
        """Run one engine. Default is PaddleOCR (normal text)."""
        kind: EngineKind = engine or "paddle"
        self._mode = kind

        try:
            if kind == "rapid":
                self.last_engine_name = "RapidOCR (ONNX)"
                return [self._rapid_predict(image)]

            if kind == "structure_table":
                self.last_engine_name = "PP-StructureV3 Table Pipeline"
                return self._structure_table_predict(image)

            # paddle (normal text)
            self.last_engine_name = "PaddleOCR"
            return self._paddle_predict(image)

        except Exception as exc:
            raise OCREngineError(f"OCR inference failed ({kind}): {exc}") from exc

    # ------------------------------------------------------------------
    # PaddleOCR — primary text engine (GPU when available)
    # ------------------------------------------------------------------
    def _ensure_paddle(self) -> Any:
        if self._paddle is not None:
            return self._paddle

        from paddleocr import PaddleOCR

        language = _paddle_language(self.config.language)
        device = self._resolved_device

        paddle_kwargs: dict[str, Any] = {
            "lang": language,
            "device": device,
            "ocr_version": (
                self.config.ocr_version
                if self.config.ocr_version
                in {"PP-OCRv3", "PP-OCRv4", "PP-OCRv5", "PP-OCRv6"}
                else "PP-OCRv5"
            ),
            # Fast path: mobile models only — never server_det / latin_rec / doc_ori.
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "PP-OCRv5_mobile_rec",
        }

        logger.info(
            "Initializing PaddleOCR (lang=%s, device=%s, det=mobile, rec=mobile)",
            language,
            device,
        )
        self._paddle = PaddleOCR(**paddle_kwargs)
        self.engine_name = "PaddleOCR"
        logger.info("PaddleOCR initialized on %s", device)
        return self._paddle

    def _paddle_predict(self, image: np.ndarray) -> list[Any]:
        pipeline = self._ensure_paddle()
        try:
            output = pipeline.predict(image)
        except TypeError:
            output = pipeline.ocr(image)
        return list(output or [])

    # ------------------------------------------------------------------
    # RapidOCR — after PaddleOCR Bad / predict failure
    # ------------------------------------------------------------------
    def _ensure_rapid(self) -> Any:
        if self._rapid is not None:
            return self._rapid

        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise OCREngineError(
                "RapidOCR fallback requested but rapidocr-onnxruntime is not installed."
            ) from exc

        use_gpu = self._resolved_device.startswith("gpu")
        rapid_kwargs: dict[str, Any] = {}
        if use_gpu:
            # onnxruntime-gpu when installed; ignored safely on CPU builds.
            rapid_kwargs = {
                "det_use_cuda": True,
                "cls_use_cuda": True,
                "rec_use_cuda": True,
            }

        try:
            self._rapid = RapidOCR(**rapid_kwargs) if rapid_kwargs else RapidOCR()
        except TypeError:
            self._rapid = RapidOCR()
        except Exception:
            logger.warning("RapidOCR GPU init failed; retrying CPU RapidOCR.")
            self._rapid = RapidOCR()

        logger.info("RapidOCR initialized (prefer_gpu=%s)", use_gpu)
        return self._rapid

    def _rapid_predict(self, image: np.ndarray) -> dict[str, Any]:
        if not self.config.enable_rapid_fallback:
            raise OCREngineError("RapidOCR fallback is disabled.")

        engine = self._ensure_rapid()
        result, _ = engine(image)

        texts: list[str] = []
        scores: list[float] = []
        polys: list[Any] = []

        for row in result or []:
            if not row or len(row) < 3:
                continue
            box, text, score = row[0], row[1], row[2]
            if not text:
                continue
            texts.append(str(text))
            scores.append(float(score))
            polys.append(box)

        return {
            "rec_texts": texts,
            "rec_scores": scores,
            "dt_polys": polys,
        }

    # ------------------------------------------------------------------
    # PP-StructureV3 Table Pipeline — only when tables detected/required
    # Avoids: server_det/rec, latin_PP-OCRv5_mobile_rec, PP-LCNet, DocLayout+
    # ------------------------------------------------------------------
    def _ensure_structure_table(self) -> Any:
        if self._structure is not None:
            return self._structure

        device = self._resolved_device
        language = _paddle_language(self.config.language)

        # Prefer dedicated table pipeline (no DocLayout_plus-L / server OCR stack).
        try:
            from paddleocr import TableRecognitionPipelineV2  # type: ignore

            self._structure = TableRecognitionPipelineV2(device=device)
            self._structure_backend = "table_v2"
            logger.info("TableRecognitionPipelineV2 initialized on %s", device)
            return self._structure
        except Exception as exc:
            logger.debug("TableRecognitionPipelineV2 unavailable: %s", exc)

        try:
            from paddlex import create_pipeline  # type: ignore

            self._structure = create_pipeline(
                pipeline="table_recognition_v2",
                device=device,
            )
            self._structure_backend = "table_v2"
            logger.info("paddlex table_recognition_v2 initialized on %s", device)
            return self._structure
        except Exception as exc:
            logger.debug("paddlex table_recognition_v2 unavailable: %s", exc)

        # Last resort: PPStructureV3 with table-only flags; never force stopped models.
        try:
            from paddleocr import PPStructureV3

            kwargs: dict[str, Any] = {
                "lang": language,
                "device": device,
                "use_doc_orientation_classify": False,  # no PP-LCNet_x1_0_doc_ori
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "use_seal_recognition": False,
                "use_table_recognition": True,
                "use_formula_recognition": False,
                "use_chart_recognition": False,
                # Prefer mobile over server_det / server_rec.
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "text_recognition_model_name": "PP-OCRv5_mobile_rec",
            }
            if self.config.pipeline_config:
                kwargs["paddlex_config"] = self.config.pipeline_config

            self._structure = PPStructureV3(**kwargs)
            self._structure_backend = "structure_v3"
            logger.info(
                "PPStructureV3 table-only initialized on %s "
                "(orientation/DocLayout+/server/latin models not forced)",
                device,
            )
            return self._structure
        except Exception as exc:
            raise OCREngineError(
                f"Failed to initialize table pipeline: {exc}"
            ) from exc

    def _structure_table_predict(self, image: np.ndarray) -> list[Any]:
        if not self.config.enable_tables:
            raise OCREngineError("Table recognition is disabled.")

        pipeline = self._ensure_structure_table()

        if self._structure_backend == "table_v2":
            try:
                output = pipeline.predict(input=image)
            except TypeError:
                try:
                    output = pipeline.predict(image)
                except TypeError:
                    output = pipeline(image)
            return list(output) if not isinstance(output, dict) else [output]

        # structure_v3 table-only
        try:
            output = pipeline.predict(
                input=image,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                use_seal_recognition=False,
                use_table_recognition=True,
                use_formula_recognition=False,
                use_chart_recognition=False,
            )
        except TypeError:
            output = pipeline.predict(input=image)
        return list(output)

    @staticmethod
    def result_to_dict(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result

        for method_name in ("to_dict", "json", "to_json"):
            method = getattr(result, method_name, None)
            if callable(method):
                try:
                    value = method()
                    if isinstance(value, str):
                        value = json.loads(value)
                    if isinstance(value, dict):
                        return value
                except Exception:
                    pass

        value = getattr(result, "res", None)
        if isinstance(value, dict):
            return value

        if isinstance(result, list):
            return _normalize_classic_ocr_list(result)

        if hasattr(result, "__dict__"):
            return {
                k: v
                for k, v in vars(result).items()
                if not k.startswith("_")
            }

        return {}


_ENGINE_CACHE: dict[tuple[Any, ...], PaddleStructureEngine] = {}
_ENGINE_CACHE_LOCK = threading.Lock()


def _cache_key(config: OCRConfig) -> tuple[Any, ...]:
    return (
        config.language,
        config.device,
        config.pipeline_name,
        config.enable_doc_orientation,
        config.enable_doc_unwarping,
        config.enable_textline_orientation,
        config.model_dir,
        config.pipeline_config,
        config.enable_seal_recognition,
        config.enable_tables,
        config.enable_rapid_fallback,
        config.ocr_version,
    )


def get_shared_engine(config: OCRConfig) -> PaddleStructureEngine:
    """Process-wide cached engine (weights load once per config)."""
    key = _cache_key(config)
    with _ENGINE_CACHE_LOCK:
        engine = _ENGINE_CACHE.get(key)
        if engine is None:
            engine = PaddleStructureEngine(config)
            _ENGINE_CACHE[key] = engine
        return engine


def _normalize_classic_ocr_list(rows: list[Any]) -> dict[str, Any]:
    texts: list[str] = []
    scores: list[float] = []
    polys: list[Any] = []

    for item in rows:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        box, pair = item[0], item[1]
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            texts.append(str(pair[0]))
            scores.append(float(pair[1]))
            polys.append(box)

    return {
        "rec_texts": texts,
        "rec_scores": scores,
        "dt_polys": polys,
    }


def _paddle_language(language: str) -> str:
    value = (
        (language or "")
        .strip()
        .casefold()
        .replace("_", "-")
    )
    if value in {"tr", "tr-tr", "turkish", "türkçe", "turkce"}:
        return "tr"
    return value or "tr"
