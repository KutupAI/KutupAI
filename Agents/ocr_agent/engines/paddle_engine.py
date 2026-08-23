"""OCR engines: PP-StructureV3 → PaddleOCR → RapidOCR (runtime fallback)."""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

import numpy as np

from Agents.ocr_agent.config import OCRConfig
from Agents.ocr_agent.device import resolve_device
from Agents.ocr_agent.exceptions import OCREngineError

# Disable oneDNN/MKLDNN early (PaddleX may enable them on CPU otherwise).
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

logger = logging.getLogger(__name__)


class PaddleStructureEngine:
    """Cached OCR engine: StructureV3, then PaddleOCR, then RapidOCR."""

    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._pipeline: Any = None
        self._mode = "structure"  # structure | ocr | rapid
        self._rapid: Any = None
        self.engine_name = "PaddleOCR + PP-StructureV3"
        self.last_engine_name = self.engine_name
        self._resolved_device = resolve_device(config.device)

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        device = self._resolved_device
        language = _paddle_language(self.config.language)
        logger.info(
            "Initializing OCR engine (lang=%s, device=%s, profile=%s)",
            language,
            device,
            self.config.performance_profile,
        )

        structure_error: Exception | None = None
        try:
            from paddleocr import PPStructureV3

            kwargs: dict[str, Any] = {
                "lang": language,
                "device": device,
                "enable_mkldnn": False,
                "ocr_version": (
                    self.config.ocr_version
                    if str(self.config.ocr_version).startswith("PP-OCR")
                    else "PP-OCRv5"
                ),
                "use_doc_orientation_classify": self.config.enable_doc_orientation,
                "use_doc_unwarping": self.config.enable_doc_unwarping,
                "use_textline_orientation": self.config.enable_textline_orientation,
                "use_seal_recognition": self.config.enable_seal_recognition,
                "use_table_recognition": self.config.enable_tables,
                "use_formula_recognition": False,
                "use_chart_recognition": False,
            }
            if language == "tr":
                kwargs["text_recognition_model_name"] = "latin_PP-OCRv5_mobile_rec"
            if self.config.pipeline_config:
                kwargs["paddlex_config"] = self.config.pipeline_config

            self._pipeline = PPStructureV3(**kwargs)
            self._mode = "structure"
            self.engine_name = "PaddleOCR + PP-StructureV3"
            self.last_engine_name = self.engine_name
            logger.info("PP-StructureV3 initialized")
            return self._pipeline
        except Exception as exc:
            structure_error = exc
            logger.warning("PP-StructureV3 init failed, falling back to PaddleOCR: %s", exc)

        try:
            from paddleocr import PaddleOCR

            paddle_kwargs: dict[str, Any] = {
                "lang": language,
                "device": device,
                "enable_mkldnn": False,
                "ocr_version": (
                    self.config.ocr_version
                    if self.config.ocr_version
                    in {"PP-OCRv3", "PP-OCRv4", "PP-OCRv5", "PP-OCRv6"}
                    else "PP-OCRv5"
                ),
                "use_doc_orientation_classify": self.config.enable_doc_orientation,
                "use_doc_unwarping": self.config.enable_doc_unwarping,
                "use_textline_orientation": self.config.enable_textline_orientation,
            }
            if language == "tr":
                paddle_kwargs["text_recognition_model_name"] = (
                    "latin_PP-OCRv5_mobile_rec"
                )

            self._pipeline = PaddleOCR(**paddle_kwargs)
            self._mode = "ocr"
            self.engine_name = "PaddleOCR (fallback)"
            self.last_engine_name = self.engine_name
            logger.info("PaddleOCR initialized")
            return self._pipeline
        except Exception as paddle_exc:
            if self.config.enable_rapid_fallback:
                logger.warning(
                    "PaddleOCR init failed, falling back to RapidOCR: %s",
                    paddle_exc,
                )
                self._pipeline = self._ensure_rapid()
                self._mode = "rapid"
                self.engine_name = "RapidOCR (ONNX fallback)"
                self.last_engine_name = self.engine_name
                return self._pipeline
            raise OCREngineError(
                "Failed to initialize OCR engines. "
                f"PP-StructureV3: {structure_error}; PaddleOCR: {paddle_exc}"
            ) from paddle_exc

    def _ensure_rapid(self) -> Any:
        if self._rapid is not None:
            return self._rapid
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise OCREngineError(
                "RapidOCR fallback requested but rapidocr-onnxruntime is not installed."
            ) from exc
        self._rapid = RapidOCR()
        return self._rapid

    def predict(self, image: np.ndarray) -> list[Any]:
        pipeline = self._ensure_pipeline()
        self.last_engine_name = self.engine_name
        try:
            if self._mode == "rapid":
                self.last_engine_name = "RapidOCR (ONNX fallback)"
                return [self._rapid_predict(image)]

            if self._mode == "structure":
                try:
                    output = pipeline.predict(
                        input=image,
                        use_doc_orientation_classify=self.config.enable_doc_orientation,
                        use_doc_unwarping=self.config.enable_doc_unwarping,
                        use_textline_orientation=self.config.enable_textline_orientation,
                        use_seal_recognition=self.config.enable_seal_recognition,
                        use_table_recognition=self.config.enable_tables,
                        use_formula_recognition=False,
                        use_chart_recognition=False,
                    )
                except TypeError:
                    output = pipeline.predict(input=image)
                self.last_engine_name = self.engine_name
                return list(output)

            try:
                output = pipeline.predict(image)
            except TypeError:
                output = pipeline.ocr(image)
            self.last_engine_name = self.engine_name
            return list(output or [])
        except Exception as exc:
            if self._mode != "rapid" and self.config.enable_rapid_fallback:
                try:
                    logger.warning(
                        "Paddle predict failed (%s); using RapidOCR for this page.",
                        exc,
                    )
                    self.last_engine_name = "RapidOCR (ONNX fallback)"
                    return [self._rapid_predict(image)]
                except Exception as rapid_exc:
                    raise OCREngineError(
                        f"OCR inference failed (Paddle: {exc}; RapidOCR: {rapid_exc})"
                    ) from rapid_exc
            raise OCREngineError(f"OCR inference failed: {exc}") from exc

    def _rapid_predict(self, image: np.ndarray) -> dict[str, Any]:
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
        return {"rec_texts": texts, "rec_scores": scores, "dt_polys": polys}

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
            return {k: v for k, v in vars(result).items() if not k.startswith("_")}
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
    return {"rec_texts": texts, "rec_scores": scores, "dt_polys": polys}


def _paddle_language(language: str) -> str:
    value = (language or "").strip().casefold().replace("_", "-")
    if value in {"tr", "tr-tr", "turkish", "türkçe", "turkce"}:
        return "tr"
    return value or "tr"
