"""OCR engines: PP-StructureV3 → PaddleOCR → RapidOCR (Windows-safe fallback)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from Agents.ocr_agent.config import OCRConfig
from Agents.ocr_agent.exceptions import OCREngineError

# Paddle 3.x + oneDNN on Windows often crashes at inference; force flags early.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")


class PaddleStructureEngine:
    """Long-lived OCR engine with layered fallbacks.

    Prefer PP-StructureV3, then plain PaddleOCR, then RapidOCR (ONNX) when
    Paddle inference fails (common on Windows oneDNN builds).
    """

    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._pipeline = None
        self._mode = "structure"  # structure | ocr | rapid
        self.engine_name = "PaddleOCR + PP-StructureV3"
        self._rapid = None

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        structure_error: Exception | None = None
        try:
            from paddleocr import PPStructureV3

            kwargs: dict[str, Any] = {
                "lang": self.config.language,
                "device": self.config.device,
                "use_doc_orientation_classify": self.config.enable_doc_orientation,
                "use_doc_unwarping": self.config.enable_doc_unwarping,
                "use_textline_orientation": self.config.enable_textline_orientation,
            }
            if self.config.pipeline_config:
                kwargs["paddlex_config"] = self.config.pipeline_config
            self._pipeline = PPStructureV3(**kwargs)
            self._mode = "structure"
            self.engine_name = "PaddleOCR + PP-StructureV3"
            return self._pipeline
        except Exception as exc:
            structure_error = exc

        try:
            from paddleocr import PaddleOCR

            self._pipeline = PaddleOCR(
                lang=self.config.language,
                device=self.config.device,
                use_doc_orientation_classify=self.config.enable_doc_orientation,
                use_doc_unwarping=self.config.enable_doc_unwarping,
                use_textline_orientation=self.config.enable_textline_orientation,
            )
            self._mode = "ocr"
            self.engine_name = "PaddleOCR (fallback)"
            return self._pipeline
        except Exception as paddle_exc:
            if self.config.enable_rapid_fallback:
                self._pipeline = self._ensure_rapid()
                self._mode = "rapid"
                self.engine_name = "RapidOCR (ONNX fallback)"
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
        try:
            if self._mode == "rapid":
                return [self._rapid_predict(image)]

            if self._mode == "structure":
                try:
                    output = pipeline.predict(
                        input=image,
                        use_doc_orientation_classify=self.config.enable_doc_orientation,
                        use_doc_unwarping=self.config.enable_doc_unwarping,
                        use_textline_orientation=self.config.enable_textline_orientation,
                    )
                except TypeError:
                    output = pipeline.predict(input=image)
                return list(output)

            try:
                output = pipeline.predict(image)
            except TypeError:
                output = pipeline.ocr(image)
            return list(output or [])
        except Exception as exc:
            # Paddle often loads but dies at inference on Windows oneDNN.
            if self._mode != "rapid" and self.config.enable_rapid_fallback:
                try:
                    self._pipeline = self._ensure_rapid()
                    self._mode = "rapid"
                    self.engine_name = "RapidOCR (ONNX fallback)"
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
            # row: [box, text, score]
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
                        import json

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
