from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

from ..models import BoundingBox, OCRTextItem

# Keep almost everything that OCR actually read. Low scores become
# `uncertain` later; dropping them is what truncated page.text before.
_KEEP_SCORE_FLOOR = 0.01

_PARA_START_RE = re.compile(
    r"^\s*(?:"
    r"\(\s*\d+\s*\)"
    r"|Madde\s*\d+"
    r"|MADDE\s*\d+"
    r"|Article\s*\d+"
    r")",
    re.IGNORECASE,
)


class OCRResultParser:
    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold

    def parse(self, raw: dict[str, Any], page_index: int) -> list[OCRTextItem]:
        source = _lift_ocr_fields(raw or {})
        items: list[OCRTextItem] = []

        polys = (
            source.get("dt_polys")
            or source.get("rec_polys")
            or source.get("rec_boxes")
            or source.get("text_boxes")
            or []
        )
        texts = source.get("rec_texts") or source.get("texts") or source.get("rec_text") or []
        if isinstance(texts, str):
            texts = [texts]
        scores = source.get("rec_scores") or source.get("scores") or []

        for i, text in enumerate(texts):
            text = "" if text is None else str(text)
            if not text.strip():
                continue
            score = float(scores[i]) if i < len(scores) else 0.0
            if score < _KEEP_SCORE_FLOOR and i < len(scores):
                continue
            poly = polys[i] if i < len(polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]
            try:
                bbox = self._bbox(poly)
            except Exception:
                bbox = BoundingBox([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            items.append(OCRTextItem(
                text=text,
                confidence=score if i < len(scores) else max(score, 0.5),
                bounding_box=bbox,
                page_index=page_index,
            ))

        if not items:
            nested_list = source.get("ocr_res") or source.get("text_res") or []
            for item in nested_list:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                box, pair = item[0], item[1]
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    text, score = str(pair[0]), float(pair[1])
                    if text.strip() and score >= _KEEP_SCORE_FLOOR:
                        items.append(OCRTextItem(text, score, self._bbox(box), page_index))

        if not items:
            items.extend(self._from_parsing_res(source, page_index))

        return sort_reading_order(items)

    def _from_parsing_res(self, raw: dict[str, Any], page_index: int) -> list[OCRTextItem]:
        blocks = raw.get("parsing_res_list") or raw.get("parsing_res") or []
        if isinstance(blocks, dict):
            blocks = [blocks]
        items: list[OCRTextItem] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            text = str(block.get("block_content") or block.get("content") or "").strip()
            if not text:
                continue
            coord = block.get("block_bbox") or block.get("bbox") or block.get("coordinate")
            bbox = self._bbox(coord) if coord else BoundingBox(
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            )
            score = float(block.get("score") or block.get("confidence") or 0.8)
            items.append(OCRTextItem(text, score, bbox, page_index))
        return items

    @staticmethod
    def _bbox(poly: Any) -> BoundingBox:
        arr = np.asarray(poly, dtype=float)
        if arr.shape == (4,):
            x1, y1, x2, y2 = arr.tolist()
            return BoundingBox([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return BoundingBox(arr.reshape(-1, 2).tolist())


def sort_reading_order(items: Sequence[OCRTextItem]) -> list[OCRTextItem]:
    """Top-to-bottom, left-to-right. Columns are read left-then-right."""

    if not items:
        return []
    if _all_zero_bbox(items):
        return list(items)

    columns = _split_columns(items)
    ordered: list[OCRTextItem] = []
    for column in columns:
        ordered.extend(sorted(column, key=lambda i: (_y0(i), _x0(i))))
    return ordered


def group_paragraphs(items: Sequence[OCRTextItem]) -> list[tuple[str, list[OCRTextItem]]]:
    """Rebuild paragraphs from bounding boxes. Never mixes stacked blocks."""

    if not items:
        return []
    if _all_zero_bbox(items):
        parts: list[tuple[str, list[OCRTextItem]]] = []
        for item in items:
            text = (item.corrected_text or item.text or "").strip("\n")
            if text:
                parts.append((text, [item]))
        return parts

    lines = _group_lines(items)
    if not lines:
        return []

    heights = [_line_height(line) for line in lines]
    median_h = float(np.median(np.asarray(heights))) if heights else 16.0
    para_gap = max(median_h * 0.85, 10.0)

    paragraphs: list[list[list[OCRTextItem]]] = []
    current: list[list[OCRTextItem]] = []
    prev_bottom = None
    for line in lines:
        text = _render_line(line)
        if not text:
            continue
        top = min(_y0(i) for i in line)
        new_para = False
        if not current:
            new_para = False
        elif prev_bottom is not None and (top - prev_bottom) >= para_gap:
            new_para = True
        elif _PARA_START_RE.match(text):
            new_para = True
        if new_para and current:
            paragraphs.append(current)
            current = [line]
        else:
            current.append(line)
        prev_bottom = max(_y1(i) for i in line)
    if current:
        paragraphs.append(current)

    out: list[tuple[str, list[OCRTextItem]]] = []
    for para_lines in paragraphs:
        rendered = [_render_line(line) for line in para_lines]
        text = "\n".join(t for t in rendered if t).strip()
        if not text:
            continue
        flat = [item for line in para_lines for item in line]
        out.append((text, flat))
    return out


def join_page_text(items: Sequence[OCRTextItem]) -> str:
    """Rebuild page text with natural line / paragraph breaks (no truncation)."""

    paragraphs = group_paragraphs(items)
    return "\n\n".join(text for text, _ in paragraphs if text).strip()


def _group_lines(items: Sequence[OCRTextItem]) -> list[list[OCRTextItem]]:
    """Group boxes into visual lines. Stacked (x-overlapping) boxes stay separate."""

    ordered = sort_reading_order(items)
    lines: list[list[OCRTextItem]] = []
    for item in ordered:
        text = (item.corrected_text or item.text or "").strip()
        if not text:
            continue
        placed = False
        # A numbered clause / article heading must not share a line with
        # leftover fragments from the previous paragraph.
        force_new = _PARA_START_RE.match(text) is not None
        if not force_new:
            for line in reversed(lines[-4:]):
                if _same_visual_line(item, line):
                    line.append(item)
                    placed = True
                    break
        if not placed:
            lines.append([item])

    for line in lines:
        line.sort(key=lambda i: _x0(i))
    lines.sort(key=lambda line: (min(_y0(i) for i in line), min(_x0(i) for i in line)))
    return lines


def _same_visual_line(item: OCRTextItem, line: list[OCRTextItem]) -> bool:
    x0, y0, x1, y1 = item.bounding_box.as_xyxy()
    height = max(8.0, y1 - y0)
    y_center = (y0 + y1) / 2.0

    for other in line:
        ox0, oy0, ox1, oy1 = other.bounding_box.as_xyxy()
        x_overlap = min(x1, ox1) - max(x0, ox0)
        min_w = min(max(1.0, x1 - x0), max(1.0, ox1 - ox0))
        # Same column / stacked words: never merge into one line.
        if x_overlap > 0.30 * min_w:
            return False

    ly0 = min(_y0(i) for i in line)
    ly1 = max(_y1(i) for i in line)
    line_center = (ly0 + ly1) / 2.0
    line_h = max(8.0, ly1 - ly0)
    overlap = min(y1, ly1) - max(y0, ly0)
    min_h = min(height, line_h)
    if overlap < 0.50 * min_h:
        return False
    if abs(y_center - line_center) > 0.38 * max(height, line_h):
        return False
    return True


def _split_columns(items: Sequence[OCRTextItem]) -> list[list[OCRTextItem]]:
    """Split a true two-column page; otherwise return a single reading stream."""

    xs1 = [_x1(i) for i in items]
    page_w = max(xs1) if xs1 else 0.0
    if page_w < 80 or len(items) < 12:
        return [list(items)]

    narrow = [i for i in items if (_x1(i) - _x0(i)) < 0.55 * page_w]
    if len(narrow) < 12:
        return [list(items)]

    centers = sorted((_x0(i) + _x1(i)) / 2.0 for i in narrow)
    gap, idx = 0.0, 0
    for i in range(len(centers) - 1):
        d = centers[i + 1] - centers[i]
        if d > gap:
            gap, idx = d, i
    if gap < 0.14 * page_w:
        return [list(items)]

    split_x = (centers[idx] + centers[idx + 1]) / 2.0
    left = [i for i in narrow if (_x0(i) + _x1(i)) / 2.0 < split_x]
    right = [i for i in narrow if (_x0(i) + _x1(i)) / 2.0 >= split_x]
    if len(left) < 4 or len(right) < 4:
        return [list(items)]

    left_y = (_ymin(left), _ymax(left))
    right_y = (_ymin(right), _ymax(right))
    overlap = min(left_y[1], right_y[1]) - max(left_y[0], right_y[0])
    min_span = min(left_y[1] - left_y[0], right_y[1] - right_y[0])
    if min_span <= 0 or overlap < 0.45 * min_span:
        return [list(items)]

    wide = [i for i in items if i not in left and i not in right]
    # Interleave full-width items by vertical position with left-then-right bands.
    return _interleave_full_width(wide, left, right)


def _interleave_full_width(
    wide: list[OCRTextItem],
    left: list[OCRTextItem],
    right: list[OCRTextItem],
) -> list[list[OCRTextItem]]:
    """Read full-width rows in place; two-column bands as left then right."""

    if not wide:
        return [left, right]

    bands: list[list[OCRTextItem]] = []
    remaining_left = list(left)
    remaining_right = list(right)
    for item in sorted(wide, key=_y0):
        y = (_y0(item) + _y1(item)) / 2.0
        before_left = [i for i in remaining_left if _y1(i) <= y]
        before_right = [i for i in remaining_right if _y1(i) <= y]
        remaining_left = [i for i in remaining_left if i not in before_left]
        remaining_right = [i for i in remaining_right if i not in before_right]
        if before_left:
            bands.append(before_left)
        if before_right:
            bands.append(before_right)
        bands.append([item])
    if remaining_left:
        bands.append(remaining_left)
    if remaining_right:
        bands.append(remaining_right)
    return bands or [left, right]


def _render_line(line: Sequence[OCRTextItem]) -> str:
    parts = [(i.corrected_text or i.text or "").strip() for i in line]
    text = " ".join(p for p in parts if p)
    return " ".join(text.split())


def _line_height(line: Sequence[OCRTextItem]) -> float:
    return max(8.0, max(_y1(i) for i in line) - min(_y0(i) for i in line))


def _all_zero_bbox(items: Sequence[OCRTextItem]) -> bool:
    for item in items:
        x0, y0, x1, y1 = item.bounding_box.as_xyxy()
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.0:
            return False
    return True


def _x0(item: OCRTextItem) -> float:
    return item.bounding_box.as_xyxy()[0]


def _y0(item: OCRTextItem) -> float:
    return item.bounding_box.as_xyxy()[1]


def _x1(item: OCRTextItem) -> float:
    return item.bounding_box.as_xyxy()[2]


def _y1(item: OCRTextItem) -> float:
    return item.bounding_box.as_xyxy()[3]


def _ymin(items: Sequence[OCRTextItem]) -> float:
    return min(_y0(i) for i in items)


def _ymax(items: Sequence[OCRTextItem]) -> float:
    return max(_y1(i) for i in items)


def _lift_ocr_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """PP-StructureV3 often nests OCR under overall_ocr_res."""

    lifted = dict(raw)
    nested = raw.get("overall_ocr_res") or raw.get("ocr_res")
    if isinstance(nested, dict):
        for key in (
            "dt_polys", "rec_polys", "rec_boxes", "text_boxes",
            "rec_texts", "texts", "rec_text", "rec_scores", "scores",
        ):
            if key not in lifted or not lifted.get(key):
                if nested.get(key):
                    lifted[key] = nested[key]
    return lifted
