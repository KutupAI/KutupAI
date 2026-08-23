from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import mimetypes

from Agents.ocr_agent.exceptions import UnsupportedDocumentError, DocumentReadError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp", ".gif"}
PDF_EXTENSIONS = {".pdf"}
OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS | OFFICE_EXTENSIONS


@dataclass(frozen=True)
class DocumentInput:
    path: Path
    file_name: str
    extension: str
    mime_type: str
    size_bytes: int

    @property
    def is_pdf(self) -> bool:
        return self.extension == ".pdf"

    @property
    def is_image(self) -> bool:
        return self.extension in IMAGE_EXTENSIONS

    @property
    def is_office(self) -> bool:
        return self.extension in OFFICE_EXTENSIONS


def validate_document(path: str | Path, max_file_size_mb: int | None = None) -> DocumentInput:
    """Validate a document path is safe and supported.

    Security: this only ever *reads* the path that Orchestration already
    resolved (Application writes uploads under `Storage/files/uploads/`);
    it never executes the file and never trusts extension alone for a
    security decision beyond "which reader to use".

    ``max_file_size_mb=None`` means no size rejection (extract everything).
    """
    p = Path(path)
    try:
        resolved = p.resolve(strict=True)
    except FileNotFoundError as exc:
        raise DocumentReadError(f"Document does not exist: {p}") from exc
    if not resolved.is_file():
        raise DocumentReadError(f"Document does not exist: {p}")

    ext = resolved.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"Unsupported document format: {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    size = resolved.stat().st_size
    if size <= 0:
        raise DocumentReadError(f"Document is empty: {resolved}")
    if max_file_size_mb is not None and size > max_file_size_mb * 1024 * 1024:
        raise UnsupportedDocumentError(
            f"Document exceeds configured limit of {max_file_size_mb} MB."
        )
    mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    return DocumentInput(resolved, resolved.name, ext, mime, size)
