from __future__ import annotations

from typing import Any
import numpy as np

from ..models import BoundingBox, OCRTextItem


class OCRResultParser:
    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold

    def parse(self, raw: dict[str, Any], page_index: int) -> list[OCRTextItem]:
        items: list[OCRTextItem] = []

        # PP-OCRv5 commonly exposes dt_polys + rec_texts + rec_scores.
        polys = raw.get("dt_polys") or raw.get("rec_polys") or raw.get("text_boxes") or []
        texts = raw.get("rec_texts") or raw.get("texts") or []
        scores = raw.get("rec_scores") or raw.get("scores") or []

        for i, text in enumerate(texts):
            text = "" if text is None else str(text)
            score = float(scores[i]) if i < len(scores) else 0.0
            poly = polys[i] if i < len(polys) else None
            if not poly or score < self.confidence_threshold or not text.strip():
                continue
            items.append(OCRTextItem(
                text=text,
                confidence=score,
                bounding_box=self._bbox(poly),
                page_index=page_index,
            ))

        # Compatibility with older nested OCR result structures.
        if not items:
            for item in raw.get("ocr_res", []) or raw.get("text_res", []) or []:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                box, pair = item[0], item[1]
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    text, score = str(pair[0]), float(pair[1])
                    if score >= self.confidence_threshold and text.strip():
                        items.append(OCRTextItem(text, score, self._bbox(box), page_index))
        return items

    @staticmethod
    def _bbox(poly: Any) -> BoundingBox:
        arr = np.asarray(poly, dtype=float)
        if arr.shape == (4,):
            x1, y1, x2, y2 = arr.tolist()
            return BoundingBox([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return BoundingBox(arr.reshape(-1, 2).tolist())
