from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import mimetypes


SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}


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


def validate_document(path: str | Path, max_file_size_mb: int) -> DocumentInput:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Document does not exist: {p}")
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document format: {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    size = p.stat().st_size
    if size > max_file_size_mb * 1024 * 1024:
        raise ValueError(f"Document exceeds configured limit of {max_file_size_mb} MB.")
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return DocumentInput(p, p.name, ext, mime, size)
