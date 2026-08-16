from __future__ import annotations

from typing import Any
import numpy as np

from ..models import BoundingBox, LayoutElement, VisualElement


_VISUAL_LABELS = {
    "image", "figure", "chart", "seal", "stamp", "signature",
    "handwriting", "formula", "header_image", "footer_image",
}


class LayoutAnalyzer:
    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold

    def analyze(self, raw: dict[str, Any], page_index: int) -> tuple[list[LayoutElement], list[VisualElement]]:
        layouts: list[LayoutElement] = []
        visuals: list[VisualElement] = []

        boxes = raw.get("layout_det_res", {}).get("boxes", [])
        for item in boxes or []:
            label = str(item.get("label", item.get("type", "unknown"))).lower()
            score = float(item.get("score", item.get("confidence", 0.0)) or 0.0)
            coord = item.get("coordinate", item.get("bbox"))
            if not coord:
                continue
            bbox = self._bbox(coord)
            if score < self.confidence_threshold:
                continue
            element = LayoutElement(label, score, bbox, page_index, metadata={"raw": self._safe_metadata(item)})
            layouts.append(element)
            if label in _VISUAL_LABELS or any(k in label for k in ("seal", "stamp", "sign", "hand")):
                visuals.append(VisualElement(label, score, bbox, page_index, "pp-structurev3"))

        # Seal recognition may be returned separately from layout detection.
        seal_res = raw.get("seal_res") or raw.get("seal_rec_res") or {}
        for item in seal_res.get("boxes", []) if isinstance(seal_res, dict) else []:
            coord = item.get("coordinate", item.get("bbox"))
            if coord:
                score = float(item.get("score", 0.0) or 0.0)
                if score >= self.confidence_threshold:
                    visuals.append(
                        VisualElement(
                            "stamp",
                            score,
                            self._bbox(coord),
                            page_index,
                            "pp-structurev3-seal-recognition",
                        )
                    )

        return layouts, visuals

    @staticmethod
    def _bbox(coord: Any) -> BoundingBox:
        arr = np.asarray(coord, dtype=float)
        if arr.shape == (4,):
            x1, y1, x2, y2 = arr.tolist()
            return BoundingBox([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return BoundingBox(arr.reshape(-1, 2).tolist())

    @staticmethod
    def _safe_metadata(value: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in value.items() if k not in {"image", "img", "crop"}}
