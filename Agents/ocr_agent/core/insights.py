"""Extract structured document insights from OCR text + visual cues."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


_DATE_RE = re.compile(
    r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})\b"
)
_ARTICLE_RE = re.compile(
    r"(?:^|\n)\s*(?:MADDE|Madde|مادة)\s*([0-9]+)\s*[-–—:.]?\s*",
    re.MULTILINE,
)
_SIGNATURE_LABEL_RE = re.compile(
    r"(?im)^\s*(İmza|Imza|İMZA|توقيع)\s*[:：]?\s*$"
)
_SIGNATURE_INLINE_RE = re.compile(
    r"(?im)\b(İmza|Imza|توقيع)\s*[:：]\s*(\S.+)$"
)
_HANDWRITING_HINT_RE = re.compile(
    r"(?i)\b(el yazıs[ıi]|handwrit|imza\s*kontrol|islak\s*imza)\b"
)


@dataclass
class ArticleBlock:
    number: str
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


@dataclass
class DocumentInsights:
    has_signature: bool = False
    has_handwritten_signature: bool = False
    signature_names: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    primary_date: str | None = None
    has_articles: bool = False
    articles: list[ArticleBlock] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["articles"] = [
            {"number": a.number, "lines": a.lines, "text": a.text} for a in self.articles
        ]
        return data


def split_lines(text: str) -> list[str]:
    """Normalize text into clean display lines (no literal \\n leftovers)."""
    if not text:
        return []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # If text was double-escaped somehow, unescape once.
    if "\\n" in normalized and "\n" not in normalized.strip():
        normalized = normalized.replace("\\n", "\n")
    lines = [ln.strip() for ln in normalized.split("\n")]
    return [ln for ln in lines if ln]


def extract_dates(text: str) -> list[str]:
    found = _DATE_RE.findall(text or "")
    # preserve order, unique
    out: list[str] = []
    for d in found:
        if d not in out:
            out.append(d)
    return out


def split_articles(text: str) -> list[ArticleBlock]:
    if not text:
        return []
    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        return []

    articles: list[ArticleBlock] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        articles.append(
            ArticleBlock(number=match.group(1), lines=split_lines(body))
        )
    return articles


def detect_signature(
    text: str,
    visual_elements: list[dict[str, Any]] | None = None,
) -> tuple[bool, bool, list[str]]:
    """
    Returns (has_signature, has_handwritten_signature, signature_names).
    """
    names: list[str] = []
    has_sig = False
    handwritten = False

    for ve in visual_elements or []:
        label = str(ve.get("element_type", "")).lower()
        if any(k in label for k in ("signature", "sign", "handwrit", "imza")):
            has_sig = True
            if "hand" in label or "yaz" in label:
                handwritten = True

    for m in _SIGNATURE_INLINE_RE.finditer(text or ""):
        has_sig = True
        name = m.group(2).strip(" .-")
        if name and name not in names:
            names.append(name)

    lines = split_lines(text or "")
    for idx, line in enumerate(lines):
        if _SIGNATURE_LABEL_RE.match(line):
            has_sig = True
            # next non-empty line often the signer name
            if idx + 1 < len(lines):
                candidate = lines[idx + 1].strip(" .-")
                if candidate and not _SIGNATURE_LABEL_RE.match(candidate):
                    if candidate not in names:
                        names.append(candidate)
                    # Name after İmza label ⇒ treat as handwritten/wet signature block
                    handwritten = True

    if _HANDWRITING_HINT_RE.search(text or "") and has_sig:
        handwritten = True

    # Official letter closings with a person name under İmza are signatures
    if has_sig and names:
        handwritten = True

    return has_sig, handwritten, names


def build_insights(
    text: str,
    visual_elements: list[dict[str, Any]] | None = None,
) -> DocumentInsights:
    lines = split_lines(text)
    dates = extract_dates(text)
    articles = split_articles(text)
    has_sig, handwritten, names = detect_signature(text, visual_elements)

    return DocumentInsights(
        has_signature=has_sig,
        has_handwritten_signature=handwritten,
        signature_names=names,
        dates=dates,
        primary_date=dates[0] if dates else None,
        has_articles=bool(articles),
        articles=articles,
        lines=lines,
    )
