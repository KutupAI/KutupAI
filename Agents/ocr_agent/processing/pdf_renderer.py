"""PDF helpers: rasterize pages and extract embedded text when available."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _open_fitz():
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz  # legacy package name
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required for PDF support.") from exc
    return fitz


class PDFRenderer:
    def __init__(self, dpi: int = 200, max_pages: int | None = None) -> None:
        self.dpi = dpi
        # None / <=0 => process every page (no hard cap).
        self.max_pages = max_pages if (max_pages is not None and max_pages > 0) else None

    def extract_text_pages(self, path: Path) -> list[str]:
        """Return native PDF text per page (empty string if page is image-only)."""
        fitz = _open_fitz()
        pages: list[str] = []
        with fitz.open(path) as doc:
            self._enforce_page_cap(len(doc))
            for page in doc:
                pages.append((page.get_text("text") or "").strip())
        return pages

    def render(self, path: Path) -> list[np.ndarray]:
        fitz = _open_fitz()
        pages: list[np.ndarray] = []
        scale = self.dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        with fitz.open(path) as doc:
            self._enforce_page_cap(len(doc))
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8)
                arr = arr.reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    arr = arr[:, :, :3]
                pages.append(arr[:, :, ::-1].copy())  # RGB -> BGR
        return pages

    def _enforce_page_cap(self, page_count: int) -> None:
        if self.max_pages is not None and page_count > self.max_pages:
            raise ValueError(
                f"PDF contains {page_count} pages; configured maximum is {self.max_pages}."
            )
