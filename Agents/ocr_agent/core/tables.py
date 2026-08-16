from __future__ import annotations

from typing import Any
import re

from ..models import BoundingBox, TableCell, TableResult


class TableExtractor:
    def extract(self, raw: dict[str, Any], page_index: int) -> list[TableResult]:
        results: list[TableResult] = []
        candidates = (
            raw.get("table_res_list")
            or raw.get("table_res")
            or raw.get("table_results")
            or []
        )
        if isinstance(candidates, dict):
            candidates = [candidates]

        for table in candidates:
            if not isinstance(table, dict):
                continue
            html = table.get("pred_html") or table.get("html")
            markdown = table.get("markdown")
            score = table.get("score") or table.get("confidence")
            bbox_raw = table.get("bbox") or table.get("coordinate")
            bbox = None
            if bbox_raw:
                try:
                    bbox = BoundingBox([
                        [float(bbox_raw[0]), float(bbox_raw[1])],
                        [float(bbox_raw[2]), float(bbox_raw[1])],
                        [float(bbox_raw[2]), float(bbox_raw[3])],
                        [float(bbox_raw[0]), float(bbox_raw[3])],
                    ])
                except Exception:
                    bbox = None

            cells = self._parse_html_cells(html)
            results.append(
                TableResult(
                    page_index=page_index,
                    bounding_box=bbox,
                    cells=cells,
                    html=html,
                    markdown=markdown,
                    confidence=float(score) if score is not None else None,
                )
            )
        return results

    @staticmethod
    def _parse_html_cells(html: str | None) -> list[TableCell]:
        if not html:
            return []
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.I | re.S)
        cells: list[TableCell] = []
        for row_idx, row in enumerate(rows):
            values = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.I | re.S)
            for col_idx, value in enumerate(values):
                text = re.sub(r"<[^>]+>", " ", value)
                text = re.sub(r"\s+", " ", text).strip()
                cells.append(TableCell(row_idx, col_idx, text))
        return cells
